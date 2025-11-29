"""Integrated teaching interface with live simulation."""

import time
from typing import Optional, Dict, Any, List
import pygame
import numpy as np

from primordial.interface.config import UIConfig
from primordial.interface.renderer import Renderer
from primordial.interface.input_handler import InputHandler
from primordial.interface.teaching_signals import TeachingSignalQueue, TeachingSignalType
from primordial.interface.demo_mode import DemonstrationController, DemoAction
from primordial.interface.audio_capture import AudioCapture

from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig
from primordial.simulation.agent_database import AgentDatabase


class IntegratedTeachingApp:
    """Teaching interface connected to a live simulation."""

    def __init__(self, ui_config: UIConfig, sim_config: Optional[SimulationConfig] = None):
        """Initialize integrated app with simulation."""
        self.ui_config = ui_config
        self.sim_config = sim_config or SimulationConfig(
            world_width=640,
            world_height=480,
            max_agents=5,
            predator_count=2,
            initial_food=30,
            learning_enabled=True,
            render_enabled=False,
        )

        self.running = False

        # Create simulation
        self.simulation = Simulation(self.sim_config)

        # UI components
        self.renderer = Renderer(ui_config)
        self.input_handler = InputHandler(ui_config)
        self.signal_queue = TeachingSignalQueue()
        self.demo_controller = DemonstrationController()
        self.audio_capture = AudioCapture(
            sample_rate=ui_config.audio_sample_rate,
            channels=ui_config.audio_channels,
            buffer_size=ui_config.waveform_history
        )

        # Stats
        self.stats = {
            "rewards": 0,
            "punishments": 0,
            "demonstrations": 0,
            "voice_samples": 0,
            "loss": 0.0
        }

        self.start_time = 0.0

        # Selected agent for stats display (None = show first agent)
        self.selected_agent_id: Optional[str] = None

        # Time scale (1.0 = normal, 2.0 = 2x speed, 0.5 = half speed)
        self.time_scale = 1.0
        self.time_scale_min = 0.25
        self.time_scale_max = 4.0

        # Agent database for saving/loading trained agents
        self.agent_db = AgentDatabase()

    def start(self) -> None:
        """Start the application."""
        self.running = True
        self.start_time = time.time()
        self.audio_capture.start()
        self.renderer.set_recording(True)

        # Auto-load best agents from previous sessions
        self._auto_load_agents()

    def stop(self) -> None:
        """Stop the application."""
        self.running = False
        self.audio_capture.stop()
        self.renderer.set_recording(False)

        # Auto-save all living agents before closing
        self._auto_save_agents()

    def _auto_save_agents(self) -> None:
        """Auto-save all living agents to the database on exit."""
        saved = 0
        for wrapper in self.simulation.agents.values():
            if wrapper.agent.is_alive:
                self.agent_db.save_agent(wrapper)
                saved += 1

        if saved > 0:
            print("\n" + "=" * 50)
            print(f"  AUTO-SAVED {saved} agents to database")
            print("  Progress will continue next session!")
            print("=" * 50 + "\n")

    def _auto_load_agents(self) -> None:
        """Auto-load the best agents from previous sessions.

        Uses composite fitness score to select agents:
        - 40% total time alive
        - 30% offspring count
        - 20% food eaten
        - 10% generation
        """
        stats = self.agent_db.get_stats()
        if stats['total_agents'] == 0:
            print("\n" + "=" * 50)
            print("  NEW SIMULATION - No saved agents found")
            print("  Starting with fresh agents")
            print("=" * 50 + "\n")
            return

        loaded = self.agent_db.auto_load_best_agents(self.simulation)

        if loaded > 0:
            print("\n" + "=" * 50)
            print(f"  CONTINUING FROM PREVIOUS SESSION")
            print("=" * 50)
            print(f"  Loaded {loaded} best agents from database")
            print(f"  Total agents in DB: {stats['total_agents']}")
            print(f"  Best lifetime: {stats['max_longest_life']:.1f}s")
            print(f"  Most offspring: {stats['max_offspring']}")
            print(f"  Highest generation: {stats['max_generation']}")
            print("=" * 50 + "\n")

    def _get_world_state(self) -> Dict[str, Any]:
        """Extract world state for rendering."""
        entities = []

        # Add agents
        for agent_id, wrapper in self.simulation.agents.items():
            if wrapper.agent.is_alive:
                entities.append({
                    "type": "agent",
                    "position": (wrapper.agent.position.x, wrapper.agent.position.y),
                    "radius": wrapper.agent.radius,
                    "angle": wrapper.agent.angle,
                    "is_eating": wrapper.agent.is_eating,
                })

        # Add food
        for food in self.simulation.world.food_items:
            if food.is_active:
                entities.append({
                    "type": "food",
                    "position": (food.position.x, food.position.y),
                    "radius": food.radius,
                })

        # Add predators
        for predator in self.simulation.world.predators:
            if predator.is_active:
                # Get predator angle from velocity if available
                angle = 0
                if hasattr(predator, 'velocity') and predator.velocity.magnitude() > 0.1:
                    import math
                    angle = math.atan2(predator.velocity.y, predator.velocity.x)
                entities.append({
                    "type": "predator",
                    "position": (predator.position.x, predator.position.y),
                    "radius": predator.radius,
                    "angle": angle,
                })

        # Add vegetation
        for veg in self.simulation.world.vegetation:
            entities.append({
                "type": "vegetation",
                "position": (veg.position.x, veg.position.y),
                "radius": veg.radius,
            })

        # Add water bodies
        from primordial.world.entities import Water
        for entity in self.simulation.world.static_entities:
            if isinstance(entity, Water):
                entities.append({
                    "type": "water",
                    "position": (entity.position.x, entity.position.y),
                    "radius": entity.radius,
                })

        return {"entities": entities}

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get selected agent's state for display (or first agent if none selected)."""
        # Find the agent to display
        agent = None
        agent_id = None

        if self.selected_agent_id and self.selected_agent_id in self.simulation.agents:
            wrapper = self.simulation.agents[self.selected_agent_id]
            if wrapper.agent.is_alive:
                agent = wrapper.agent
                agent_id = self.selected_agent_id

        # Fallback to first living agent
        if agent is None:
            for aid, wrapper in self.simulation.agents.items():
                if wrapper.agent.is_alive:
                    agent = wrapper.agent
                    agent_id = aid
                    break

        if agent is None:
            return {"energy": 0, "health": 0, "age": 0, "survival_time": 0,
                    "agent_id": "none", "gender": "?", "breeding_drive": 0, "can_breed": False}

        return {
            "agent_id": agent_id,
            "energy": agent.energy / agent.genome.max_energy,
            "health": agent.health / agent.genome.max_health,
            "age": agent.age,
            "survival_time": agent.age,
            "gender": agent.gender.value,
            "breeding_drive": agent.breeding_drive,
            "can_breed": agent.can_breed(),
        }

    def _get_all_agents_data(self) -> List[Dict[str, Any]]:
        """Get data for all agents for the table display."""
        agents_data = []
        for agent_id, wrapper in self.simulation.agents.items():
            agent = wrapper.agent
            agents_data.append({
                'id': agent_id,
                'alive': agent.is_alive,
                'generation': wrapper.generation,
                'energy': agent.energy / agent.genome.max_energy if agent.is_alive else 0,
                'health': agent.health / agent.genome.max_health if agent.is_alive else 0,
                'age': agent.age,
                'gender': agent.gender.value if hasattr(agent, 'gender') else '?',
                'breeding_drive': agent.breeding_drive if hasattr(agent, 'breeding_drive') else 0,
                'social': agent.social_connection if hasattr(agent, 'social_connection') else 0.5,
            })
        # Sort: alive first, then by age descending
        agents_data.sort(key=lambda x: (-int(x['alive']), -x['age']))
        return agents_data

    def _get_target_agent_wrapper(self):
        """Get the wrapper for the selected agent (or first living agent if none selected)."""
        # Try selected agent first
        if self.selected_agent_id and self.selected_agent_id in self.simulation.agents:
            wrapper = self.simulation.agents[self.selected_agent_id]
            if wrapper.agent.is_alive:
                return wrapper

        # Fallback to first living agent
        for wrapper in self.simulation.agents.values():
            if wrapper.agent.is_alive:
                return wrapper

        return None

    def _process_events(self, timestamp: float) -> None:
        """Process input events."""
        mouse_pos = pygame.mouse.get_pos()
        # Convert screen pos to world pos (assuming world view starts at 0,0)
        world_x, world_y = mouse_pos[0], mouse_pos[1] - 40  # Subtract header height

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if event.type == pygame.KEYDOWN and event.key == self.ui_config.keys.QUIT:
                self.running = False
                continue

            # Environment editing controls
            if event.type == pygame.KEYDOWN:
                self._handle_edit_key(event.key, world_x, world_y)

            # Right-click to remove nearest entity
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self._remove_nearest_entity(world_x, world_y)

            # Left-click to select agent (in world or table)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if click is in the agent table area
                table_rect = self.ui_config.layout.agent_table_rect
                if (table_rect[0] <= mouse_pos[0] <= table_rect[0] + table_rect[2] and
                    table_rect[1] <= mouse_pos[1] <= table_rect[1] + table_rect[3]):
                    self._select_agent_from_table(mouse_pos[1] - table_rect[1])
                else:
                    self._select_agent_at(world_x, world_y)

            # Process teaching signals
            signals = self.input_handler.process_event(event, timestamp)

            for signal in signals:
                self.signal_queue.enqueue(signal)

                # Inject into simulation (affects selected agent only, or first if none selected)
                target_wrapper = self._get_target_agent_wrapper()

                if signal.signal_type == TeachingSignalType.REWARD:
                    self.stats["rewards"] += 1
                    if target_wrapper:
                        target_wrapper.events.append('human_reward')

                elif signal.signal_type == TeachingSignalType.PUNISH:
                    self.stats["punishments"] += 1
                    if target_wrapper:
                        target_wrapper.events.append('human_punish')

                elif signal.signal_type == TeachingSignalType.DEMONSTRATE:
                    self.stats["demonstrations"] += 1

    def _handle_edit_key(self, key: int, world_x: float, world_y: float) -> None:
        """Handle environment editing key presses."""
        from primordial.world.geometry import Vec2
        from primordial.world.entities import Food, Predator, Vegetation, Water

        pos = Vec2(world_x, world_y)

        # F = Add food at mouse position
        if key == pygame.K_f:
            food = Food(
                entity_id=self.simulation.world.next_entity_id,
                position=pos,
                energy_value=50.0,
                sound_intensity=0.1,
            )
            self.simulation.world.add_entity(food)
            print(f"Added food at ({world_x:.0f}, {world_y:.0f})")

        # V = Add vegetation at mouse position
        elif key == pygame.K_v:
            veg = Vegetation(
                entity_id=self.simulation.world.next_entity_id,
                position=pos,
                radius=20.0,
            )
            self.simulation.world.add_entity(veg)
            print(f"Added vegetation at ({world_x:.0f}, {world_y:.0f})")

        # P = Add predator at mouse position
        elif key == pygame.K_p:
            predator = Predator(
                entity_id=self.simulation.world.next_entity_id,
                position=pos,
                patrol_center=pos,
                patrol_radius=100.0,
            )
            self.simulation.world.add_entity(predator)
            print(f"Added predator at ({world_x:.0f}, {world_y:.0f})")

        # W = Add water at mouse position
        elif key == pygame.K_w:
            water = Water(
                entity_id=self.simulation.world.next_entity_id,
                position=pos,
                radius=30.0,
            )
            self.simulation.world.add_entity(water)
            print(f"Added water at ({world_x:.0f}, {world_y:.0f})")

        # D = Delete all vegetation (clear paths)
        elif key == pygame.K_d:
            for veg in list(self.simulation.world.vegetation):
                self.simulation.world.remove_entity(veg.id)
            print("Cleared all vegetation")

        # T = Reset/respawn agent with full energy (changed from R since R is reward)
        elif key == pygame.K_t:
            for wrapper in self.simulation.agents.values():
                wrapper.agent.energy = wrapper.agent.genome.max_energy
                wrapper.agent.health = wrapper.agent.genome.max_health
                print(f"Reset agent energy/health to max")

        # [ = Slow down time
        elif key == pygame.K_LEFTBRACKET:
            self.time_scale = max(self.time_scale_min, self.time_scale / 1.5)
            print(f"Time scale: {self.time_scale:.2f}x")

        # ] = Speed up time
        elif key == pygame.K_RIGHTBRACKET:
            self.time_scale = min(self.time_scale_max, self.time_scale * 1.5)
            print(f"Time scale: {self.time_scale:.2f}x")

        # \ = Reset time to normal
        elif key == pygame.K_BACKSLASH:
            self.time_scale = 1.0
            print(f"Time scale: {self.time_scale:.2f}x (reset)")

        # S = Save selected agent to database (or all if shift held)
        elif key == pygame.K_s:
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                self._save_all_agents_to_db()
            else:
                self._save_selected_agent_to_db()

        # L = List agents in database
        elif key == pygame.K_l:
            self._list_agents_in_db()

        # 1-9 = Load agent from database into selected slot
        elif pygame.K_1 <= key <= pygame.K_9:
            db_index = key - pygame.K_1  # 0-8
            self._load_agent_from_db(db_index)

        # M = Save/Load world map
        elif key == pygame.K_m:
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                self._load_world_map()
            else:
                self._save_world_map()

    def _select_agent_at(self, world_x: float, world_y: float) -> None:
        """Select the agent nearest to click position."""
        from primordial.world.geometry import Vec2

        pos = Vec2(world_x, world_y)
        nearest_id = None
        nearest_dist = float('inf')

        for agent_id, wrapper in self.simulation.agents.items():
            if not wrapper.agent.is_alive:
                continue
            dist = pos.distance_to(wrapper.agent.position)
            if dist < nearest_dist and dist < 30:  # Within 30 units of agent
                nearest_id = agent_id
                nearest_dist = dist

        if nearest_id:
            self.selected_agent_id = nearest_id
            print(f"Selected {nearest_id}")
        else:
            # Click on empty space deselects
            self.selected_agent_id = None

    def _select_agent_from_table(self, y_offset: int) -> None:
        """Select an agent by clicking on its row in the table.

        Args:
            y_offset: Y position relative to table top.
        """
        # Calculate which row was clicked
        # Header takes ~46 pixels (title + column headers + separator)
        header_height = 46
        line_height = 18

        if y_offset < header_height:
            return  # Clicked on header

        row_index = (y_offset - header_height) // line_height

        # Get sorted agents data (same order as table display)
        agents_data = self._get_all_agents_data()

        if 0 <= row_index < len(agents_data):
            agent_id = agents_data[row_index]['id']
            self.selected_agent_id = agent_id
            is_alive = agents_data[row_index]['alive']
            status = "alive" if is_alive else "DEAD"
            print(f"Selected {agent_id} ({status})")

    def _remove_nearest_entity(self, world_x: float, world_y: float) -> None:
        """Remove the nearest non-agent entity to mouse position."""
        from primordial.world.geometry import Vec2

        pos = Vec2(world_x, world_y)
        nearest = None
        nearest_dist = float('inf')

        # Check all entities except agents
        for entity in list(self.simulation.world.entities.values()):
            if entity.entity_type.name == "AGENT":
                continue
            dist = pos.distance_to(entity.position)
            if dist < nearest_dist and dist < 50:  # Within 50 units
                nearest = entity
                nearest_dist = dist

        if nearest:
            self.simulation.world.remove_entity(nearest.id)
            print(f"Removed {nearest.entity_type.name} at ({nearest.position.x:.0f}, {nearest.position.y:.0f})")

    def _inject_microphone_sound(self) -> None:
        """Inject microphone audio as a sound source near the selected agent.

        The microphone is heard by the selected agent as if you're speaking
        directly to them. The intensity is based on the current audio level.
        """
        from primordial.world.sound.sound_source import SoundSource
        import numpy as np

        # Get target agent
        wrapper = self._get_target_agent_wrapper()
        if wrapper is None:
            return

        # Get recent audio and compute intensity (RMS)
        audio = self.audio_capture.get_recent(1024)
        if len(audio) == 0:
            return

        rms = np.sqrt(np.mean(audio ** 2))
        # Scale RMS to reasonable intensity (0-1 range)
        intensity = min(1.0, rms * 5.0)  # Adjust multiplier as needed

        if intensity < 0.01:  # Skip if too quiet
            return

        # Create sound source at agent's position (they hear it directly)
        # Use a mid-range frequency for human voice (~300 Hz)
        source = SoundSource(
            position=wrapper.agent.position.copy(),
            frequency=300.0,  # Approximate human voice frequency
            intensity=intensity,
            is_active=True,
        )

        # Add to world's sound system
        self.simulation.world.sound_system.add_source(source)

    def _save_selected_agent_to_db(self) -> None:
        """Save the selected agent to the database."""
        wrapper = self._get_target_agent_wrapper()
        if wrapper is None:
            print("No agent selected to save")
            return

        agent_id = self.agent_db.save_agent(wrapper)
        stats = wrapper.lifetime_stats
        print(f"\n=== Agent Saved to Database ===")
        print(f"  Database ID: {agent_id}")
        print(f"  Name: {wrapper.agent_id}_gen{wrapper.generation}")
        print(f"  Generation: {wrapper.generation}")
        print(f"  Food eaten: {stats['total_food_eaten']}")
        print(f"  Times bred: {stats['times_bred']}")
        print(f"  Offspring: {stats['offspring_count']}")
        print(f"  Longest life: {stats['longest_life']:.1f}s")
        print(f"================================\n")

    def _save_all_agents_to_db(self) -> None:
        """Save all living agents to the database."""
        saved_count = 0
        for wrapper in self.simulation.agents.values():
            if wrapper.agent.is_alive:
                self.agent_db.save_agent(wrapper)
                saved_count += 1

        print(f"\n=== Saved {saved_count} agents to database ===")
        db_stats = self.agent_db.get_stats()
        print(f"  Total agents in DB: {db_stats['total_agents']}")
        print(f"  Best longest life: {db_stats['max_longest_life']:.1f}s")
        print(f"  Most food eaten: {db_stats['max_food_eaten']}")
        print(f"  Most offspring: {db_stats['max_offspring']}")
        print(f"=========================================\n")

    def _save_world_map(self) -> None:
        """Save the current world map (vegetation, water, etc) to a file."""
        import json
        from pathlib import Path

        maps_dir = Path.home() / ".primordial" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        map_data = {
            'width': self.simulation.world.width,
            'height': self.simulation.world.height,
            'vegetation': [],
            'water': [],
        }

        # Save vegetation
        for veg in self.simulation.world.vegetation:
            map_data['vegetation'].append({
                'x': veg.position.x,
                'y': veg.position.y,
                'radius': veg.radius,
            })

        # Save water
        from primordial.world.entities import Water
        for entity in self.simulation.world.static_entities:
            if isinstance(entity, Water):
                map_data['water'].append({
                    'x': entity.position.x,
                    'y': entity.position.y,
                    'radius': entity.radius,
                })

        # Save to file
        map_path = maps_dir / "world_map.json"
        with open(map_path, 'w') as f:
            json.dump(map_data, f, indent=2)

        print(f"\n{'='*50}")
        print(f"  WORLD MAP SAVED")
        print(f"{'='*50}")
        print(f"  Vegetation: {len(map_data['vegetation'])} items")
        print(f"  Water: {len(map_data['water'])} items")
        print(f"  Path: {map_path}")
        print(f"{'='*50}\n")

    def _load_world_map(self) -> None:
        """Load a saved world map."""
        import json
        from pathlib import Path
        from primordial.world.geometry import Vec2
        from primordial.world.entities import Vegetation, Water

        map_path = Path.home() / ".primordial" / "maps" / "world_map.json"

        if not map_path.exists():
            print(f"\nNo saved map found at {map_path}")
            print(f"Press M to save the current map first.\n")
            return

        with open(map_path, 'r') as f:
            map_data = json.load(f)

        # Clear existing vegetation and water
        for veg in list(self.simulation.world.vegetation):
            self.simulation.world.remove_entity(veg.id)

        for entity in list(self.simulation.world.static_entities):
            if isinstance(entity, Water):
                self.simulation.world.remove_entity(entity.id)

        # Load vegetation
        for veg_data in map_data.get('vegetation', []):
            veg = Vegetation(
                entity_id=self.simulation.world.next_entity_id,
                position=Vec2(veg_data['x'], veg_data['y']),
                radius=veg_data['radius'],
            )
            self.simulation.world.add_entity(veg)

        # Load water
        for water_data in map_data.get('water', []):
            water = Water(
                entity_id=self.simulation.world.next_entity_id,
                position=Vec2(water_data['x'], water_data['y']),
                radius=water_data['radius'],
            )
            self.simulation.world.add_entity(water)

        print(f"\n{'='*50}")
        print(f"  WORLD MAP LOADED")
        print(f"{'='*50}")
        print(f"  Vegetation: {len(map_data.get('vegetation', []))} items")
        print(f"  Water: {len(map_data.get('water', []))} items")
        print(f"{'='*50}\n")

    def _list_agents_in_db(self) -> None:
        """List agents in the database."""
        agents = self.agent_db.list_agents(order_by='longest_life', limit=9)

        # Count dead slots
        dead_slots = sum(1 for w in self.simulation.agents.values() if not w.agent.is_alive)

        print(f"\n{'='*60}")
        print(f"  SAVED AGENTS IN DATABASE")
        print(f"{'='*60}")
        for i, a in enumerate(agents):
            print(f"  [{i+1}] {a.name[:20]:<20} Gen:{a.generation:>3}  Life:{a.longest_life:>6.1f}s  Food:{a.total_food_eaten:>4}")
        print(f"{'='*60}")
        print(f"  Press 1-9 to load agent into a DEAD slot")
        print(f"  Dead slots available: {dead_slots}")
        if dead_slots == 0:
            print(f"  (No dead slots - wait for an agent to die first)")
        print(f"{'='*60}\n")

    def _load_agent_from_db(self, index: int) -> None:
        """Load an agent from database into a dead agent slot.

        Args:
            index: Index in the top agents list (0-8 for keys 1-9)
        """
        # Find a DEAD agent slot to load into (never replace living agents)
        wrapper = None
        for w in self.simulation.agents.values():
            if not w.agent.is_alive:
                wrapper = w
                break

        if wrapper is None:
            print(f"\n*** No dead agent slots available! ***")
            print(f"    Wait for an agent to die, then press {index + 1} again.\n")
            return

        # Get top agents from DB
        agents = self.agent_db.list_agents(order_by='longest_life', limit=9)
        if index >= len(agents):
            print(f"No agent at position {index + 1} in database")
            return

        record = agents[index]

        # Load into the dead wrapper
        if self.agent_db.load_agent_into_wrapper(record.id, wrapper):
            self.simulation.world.add_entity(wrapper.agent)
            print(f"\n{'='*50}")
            print(f"  LOADED: {record.name}")
            print(f"{'='*50}")
            print(f"  Into slot: {wrapper.agent_id}")
            print(f"  Generation: {record.generation}")
            print(f"  Best life: {record.longest_life:.1f}s")
            print(f"{'='*50}\n")
        else:
            print(f"Failed to load agent #{record.id}")

    def _update(self, dt: float) -> None:
        """Update simulation.

        Uses variable timestep - simulation runs at real time regardless
        of frame rate. Time scale allows speeding up or slowing down.
        """
        # Apply time scale to elapsed time
        scaled_dt = dt * self.time_scale
        self.simulation.tick(scaled_dt)

        # Push-to-talk: configured key unmutes microphone while held
        keys = pygame.key.get_pressed()
        if keys[self.ui_config.keys.PUSH_TO_TALK]:
            self.audio_capture.unmute()
            # Add microphone as sound source at selected agent's position
            self._inject_microphone_sound()
        else:
            self.audio_capture.mute()

        # Handle demo mode - human controls agent
        if self.input_handler.state.control_mode:
            action = DemoAction.from_key_input(
                up=keys[pygame.K_UP],
                down=keys[pygame.K_DOWN],
                left=keys[pygame.K_LEFT],
                right=keys[pygame.K_RIGHT]
            )

            if action.action_type == "move":
                for wrapper in self.simulation.agents.values():
                    if wrapper.agent.is_alive:
                        # Apply movement directly
                        dx, dy = action.move_direction
                        speed = 100.0 * dt
                        wrapper.agent.position.x += dx * speed
                        wrapper.agent.position.y += dy * speed

    def _render(self) -> None:
        """Render frame."""
        waveform = self.audio_capture.get_recent(self.ui_config.waveform_history)

        # Build mode string with mic status
        if self.input_handler.state.control_mode:
            mode = "CONTROL"
        else:
            mode = "OBSERVE"

        # Add mic indicator
        if not self.audio_capture.is_muted():
            mode += " | MIC ON"

        # Add time scale if not 1.0
        if self.time_scale != 1.0:
            mode += f" | {self.time_scale:.1f}x"

        self.renderer.render_frame(
            world_state=self._get_world_state(),
            agent_state=self._get_agent_state(),
            agent_view=None,  # Would need to render agent's POV
            waveform=waveform,
            metrics=self.stats,
            mode=mode,
            zoom=self.input_handler.state.zoom_level,
            agents_table_data=self._get_all_agents_data(),
            selected_agent_id=self.selected_agent_id
        )

    def run(self) -> None:
        """Main loop."""
        self.start()

        try:
            while self.running:
                dt = self.renderer.tick()
                current_time = time.time() - self.start_time

                self._process_events(current_time)
                self._update(dt)
                self._render()

                # Check if all agents dead - respawn
                if all(not w.agent.is_alive for w in self.simulation.agents.values()):
                    print(f"All agents died. Resetting simulation...")
                    self.simulation.reset()

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.audio_capture.stop()
        self.renderer.cleanup()


def main():
    """Run integrated teaching interface."""
    ui_config = UIConfig()
    sim_config = SimulationConfig(
        world_width=640,
        world_height=480,
        max_agents=10,
        predator_count=2,
        initial_food=30,
        learning_enabled=True,
    )

    app = IntegratedTeachingApp(ui_config, sim_config)
    app.run()


if __name__ == "__main__":
    main()
