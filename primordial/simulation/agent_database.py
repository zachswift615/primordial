"""SQLite database for storing and retrieving trained agents."""

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AgentRecord:
    """Record of a saved agent in the database."""
    id: int
    name: str
    generation: int
    total_food_eaten: int
    times_bred: int
    offspring_count: int
    deaths: int
    total_time_alive: float
    longest_life: float
    damage_taken: float
    created_at: float
    saved_at: float
    notes: str
    genome_json: str
    model_path: str
    is_favorite: int = 0  # 1 if favorited, 0 otherwise


class AgentDatabase:
    """SQLite database for managing saved agents.

    Stores agent genomes, neural network weights, and lifetime statistics.
    Allows querying and sorting agents by various metrics.
    """

    def __init__(self, db_path: str = None):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.primordial/agents.db
        """
        if db_path is None:
            db_dir = Path.home() / ".primordial"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "agents.db")

        self.db_path = db_path
        self.models_dir = Path(db_path).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                generation INTEGER DEFAULT 0,
                total_food_eaten INTEGER DEFAULT 0,
                times_bred INTEGER DEFAULT 0,
                offspring_count INTEGER DEFAULT 0,
                deaths INTEGER DEFAULT 0,
                total_time_alive REAL DEFAULT 0.0,
                longest_life REAL DEFAULT 0.0,
                damage_taken REAL DEFAULT 0.0,
                created_at REAL,
                saved_at REAL,
                notes TEXT DEFAULT '',
                genome_json TEXT,
                model_path TEXT,
                is_favorite INTEGER DEFAULT 0
            )
        ''')

        # Add is_favorite column if it doesn't exist (migration for existing DBs)
        try:
            cursor.execute('ALTER TABLE agents ADD COLUMN is_favorite INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_longest_life ON agents(longest_life DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_total_food ON agents(total_food_eaten DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_offspring ON agents(offspring_count DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_generation ON agents(generation DESC)')

        conn.commit()
        conn.close()

    def save_agent(self, wrapper, name: str = None, notes: str = "", db_id: int = None) -> int:
        """Save an agent to the database.

        If db_id is provided, updates the existing record instead of creating new.

        Args:
            wrapper: AgentWrapper instance to save.
            name: Optional name for the agent. Defaults to agent_id.
            notes: Optional notes about this agent.
            db_id: If provided, update this existing record instead of creating new.

        Returns:
            Database ID of the saved agent.
        """
        from primordial.simulation.agent_wrapper import AgentWrapper

        if name is None:
            name = f"{wrapper.agent_id}_gen{wrapper.generation}"

        # Serialize genome
        genome_json = json.dumps(wrapper.agent.genome.to_dict())

        # Get stats
        stats = wrapper.lifetime_stats

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if db_id is not None:
            # Update existing record
            existing = self.get_agent(db_id)
            if existing:
                # Update model weights in same file
                model_path = existing.model_path
                torch.save(wrapper.model.state_dict(), model_path)

                cursor.execute('''
                    UPDATE agents SET
                        name = ?, generation = ?, total_food_eaten = ?, times_bred = ?,
                        offspring_count = ?, deaths = ?, total_time_alive = ?,
                        longest_life = ?, damage_taken = ?, saved_at = ?,
                        notes = ?, genome_json = ?
                    WHERE id = ?
                ''', (
                    name,
                    wrapper.generation,
                    stats['total_food_eaten'],
                    stats['times_bred'],
                    stats['offspring_count'],
                    stats['deaths'],
                    stats['total_time_alive'],
                    stats['longest_life'],
                    stats['damage_taken'],
                    time.time(),
                    notes or existing.notes,
                    genome_json,
                    db_id
                ))
                conn.commit()
                conn.close()
                return db_id

        # Create new record
        model_filename = f"model_{int(time.time())}_{wrapper.agent_id}.pt"
        model_path = str(self.models_dir / model_filename)
        torch.save(wrapper.model.state_dict(), model_path)

        cursor.execute('''
            INSERT INTO agents (
                name, generation, total_food_eaten, times_bred, offspring_count,
                deaths, total_time_alive, longest_life, damage_taken,
                created_at, saved_at, notes, genome_json, model_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name,
            wrapper.generation,
            stats['total_food_eaten'],
            stats['times_bred'],
            stats['offspring_count'],
            stats['deaths'],
            stats['total_time_alive'],
            stats['longest_life'],
            stats['damage_taken'],
            stats.get('created_at') or time.time(),
            time.time(),
            notes,
            genome_json,
            model_path
        ))

        agent_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return agent_id

    def list_agents(self,
                    order_by: str = "saved_at",
                    descending: bool = True,
                    limit: int = 50) -> List[AgentRecord]:
        """List agents from the database.

        Args:
            order_by: Column to sort by. Options: saved_at, longest_life,
                      total_food_eaten, times_bred, offspring_count, generation
            descending: Sort in descending order if True.
            limit: Maximum number of results.

        Returns:
            List of AgentRecord objects.
        """
        valid_columns = {
            'saved_at', 'longest_life', 'total_food_eaten',
            'times_bred', 'offspring_count', 'generation', 'deaths'
        }
        if order_by not in valid_columns:
            order_by = 'saved_at'

        direction = "DESC" if descending else "ASC"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f'''
            SELECT * FROM agents
            ORDER BY {order_by} {direction}
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [AgentRecord(*row) for row in rows]

    def get_agent(self, agent_id: int) -> Optional[AgentRecord]:
        """Get a specific agent by ID.

        Args:
            agent_id: Database ID of the agent.

        Returns:
            AgentRecord or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM agents WHERE id = ?', (agent_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return AgentRecord(*row)
        return None

    def load_agent_into_wrapper(self, agent_id: int, wrapper) -> bool:
        """Load a saved agent into an existing wrapper.

        Args:
            agent_id: Database ID of the agent to load.
            wrapper: AgentWrapper to load into.

        Returns:
            True if successful, False otherwise.
        """
        from primordial.agents.genome import AgentGenome
        from primordial.agents.body import AgentBody

        record = self.get_agent(agent_id)
        if record is None:
            return False

        # Load genome
        genome_data = json.loads(record.genome_json)
        genome = AgentGenome.from_dict(genome_data)

        # Create new agent body with loaded genome
        wrapper.agent = AgentBody(
            agent_id=wrapper.agent_id,
            genome=genome,
            initial_position=wrapper.agent.position
        )

        # Load model weights
        if os.path.exists(record.model_path):
            wrapper.model.load_state_dict(torch.load(record.model_path))

        # Reset the learning loop to use the new model weights
        # This recreates the optimizer with fresh state for the loaded model
        if wrapper.learning_enabled and wrapper.learning_loop is not None:
            from primordial.learning.learning_loop import OnlineLearningLoop
            # Recreate learning loop with same config from wrapper's simulation config
            optimizer_config = {
                'type': 'adamw',
                'params': {'lr': wrapper.config.learning_rate},
                'lr_schedule': {'warmup_steps': 100}
            }
            reward_config = {
                'modulation': {'reward_scale': wrapper.config.reward_modulation_scale}
            }
            wrapper.learning_loop = OnlineLearningLoop(
                model=wrapper.model,
                optimizer_config=optimizer_config,
                reward_config=reward_config,
                agent_id=wrapper.agent_id,
            )

        # Reset sensing state to avoid stale references
        wrapper.prev_senses = None
        wrapper.prev_modalities = None
        wrapper.prev_agent_state = None

        # Restore stats
        wrapper.generation = record.generation
        wrapper.lifetime_stats = {
            'total_food_eaten': record.total_food_eaten,
            'times_bred': record.times_bred,
            'offspring_count': record.offspring_count,
            'deaths': record.deaths,
            'total_time_alive': record.total_time_alive,
            'longest_life': record.longest_life,
            'damage_taken': record.damage_taken,
            'created_at': record.created_at,
        }

        # Track database ID so we can update instead of create new
        wrapper.db_id = record.id

        return True

    def delete_agent(self, agent_id: int) -> bool:
        """Delete an agent from the database.

        Args:
            agent_id: Database ID of the agent to delete.

        Returns:
            True if deleted, False if not found.
        """
        record = self.get_agent(agent_id)
        if record is None:
            return False

        # Delete model file
        if os.path.exists(record.model_path):
            os.remove(record.model_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM agents WHERE id = ?', (agent_id,))
        conn.commit()
        conn.close()

        return True

    def toggle_favorite(self, agent_id: int) -> bool:
        """Toggle favorite status for an agent.

        Args:
            agent_id: Database ID of the agent.

        Returns:
            New favorite status (True if now favorited).
        """
        record = self.get_agent(agent_id)
        if record is None:
            return False

        new_status = 0 if record.is_favorite else 1

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE agents SET is_favorite = ? WHERE id = ?', (new_status, agent_id))
        conn.commit()
        conn.close()

        return new_status == 1

    def set_favorite(self, agent_id: int, is_favorite: bool) -> bool:
        """Set favorite status for an agent.

        Args:
            agent_id: Database ID of the agent.
            is_favorite: Whether to mark as favorite.

        Returns:
            True if successful.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE agents SET is_favorite = ? WHERE id = ?',
                      (1 if is_favorite else 0, agent_id))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def get_favorites(self) -> List[AgentRecord]:
        """Get all favorited agents.

        Returns:
            List of favorite AgentRecord objects.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM agents WHERE is_favorite = 1 ORDER BY saved_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return [AgentRecord(*row) for row in rows]

    def cleanup_duplicates(self, keep_top_n: int = 10, keep_favorites: bool = True) -> Dict[str, int]:
        """Remove duplicate/old agents, keeping only the best.

        This helps prevent database bloat from auto-saving.

        Args:
            keep_top_n: Number of top agents to keep (by fitness score).
            keep_favorites: If True, always keep favorited agents.

        Returns:
            Dict with cleanup stats (deleted_count, kept_count, etc).
        """
        # Get all agents
        all_agents = self.list_agents(order_by='saved_at', limit=10000)
        if not all_agents:
            return {'deleted_count': 0, 'kept_count': 0}

        # Get best agents by fitness
        best_agents = self.get_best_agents(limit=keep_top_n)
        best_ids = {a.id for a in best_agents}

        # Get favorites
        favorites = self.get_favorites() if keep_favorites else []
        favorite_ids = {a.id for a in favorites}

        # IDs to keep
        keep_ids = best_ids | favorite_ids

        # Delete the rest
        deleted_count = 0
        for agent in all_agents:
            if agent.id not in keep_ids:
                if self.delete_agent(agent.id):
                    deleted_count += 1

        return {
            'deleted_count': deleted_count,
            'kept_count': len(keep_ids),
            'favorites_kept': len(favorite_ids),
            'top_performers_kept': len(best_ids),
        }

    def search_agents(self, query: str) -> List[AgentRecord]:
        """Search agents by name or notes.

        Args:
            query: Search string.

        Returns:
            List of matching AgentRecord objects.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM agents
            WHERE name LIKE ? OR notes LIKE ?
            ORDER BY saved_at DESC
        ''', (f'%{query}%', f'%{query}%'))

        rows = cursor.fetchall()
        conn.close()

        return [AgentRecord(*row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics.

        Returns:
            Dictionary with stats like total agents, best performer, etc.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM agents')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT MAX(longest_life) FROM agents')
        max_life = cursor.fetchone()[0] or 0

        cursor.execute('SELECT MAX(total_food_eaten) FROM agents')
        max_food = cursor.fetchone()[0] or 0

        cursor.execute('SELECT MAX(offspring_count) FROM agents')
        max_offspring = cursor.fetchone()[0] or 0

        cursor.execute('SELECT MAX(generation) FROM agents')
        max_gen = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_agents': total,
            'max_longest_life': max_life,
            'max_food_eaten': max_food,
            'max_offspring': max_offspring,
            'max_generation': max_gen,
        }

    def format_agent_list(self, agents: List[AgentRecord]) -> str:
        """Format a list of agents for display.

        Args:
            agents: List of AgentRecord objects.

        Returns:
            Formatted string for terminal display.
        """
        if not agents:
            return "No agents found."

        lines = [
            f"{'ID':>4} | {'★':>1} | {'Name':<23} | {'Gen':>4} | {'Food':>5} | {'Bred':>4} | {'Kids':>4} | {'Longest':>8} | {'Deaths':>6}",
            "-" * 95
        ]

        for a in agents:
            fav = "★" if a.is_favorite else " "
            lines.append(
                f"{a.id:>4} | {fav:>1} | {a.name[:23]:<23} | {a.generation:>4} | "
                f"{a.total_food_eaten:>5} | {a.times_bred:>4} | {a.offspring_count:>4} | "
                f"{a.longest_life:>7.1f}s | {a.deaths:>6}"
            )

        return "\n".join(lines)

    def get_best_agents(self, limit: int = 10) -> List[AgentRecord]:
        """Get the best agents ranked by composite fitness score.

        Favorites are always included first, then filled with top performers.

        Fitness score combines:
        - Total time alive (40% weight) - survival ability
        - Offspring count (30% weight) - reproductive success
        - Total food eaten (20% weight) - foraging ability
        - Generation (10% weight) - evolutionary progress

        Args:
            limit: Maximum number of agents to return.

        Returns:
            List of AgentRecord objects sorted by fitness (favorites first).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all agents
        cursor.execute('SELECT * FROM agents')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        agents = [AgentRecord(*row) for row in rows]

        # Calculate fitness scores
        # First, find max values for normalization
        max_time = max(a.total_time_alive for a in agents) or 1
        max_offspring = max(a.offspring_count for a in agents) or 1
        max_food = max(a.total_food_eaten for a in agents) or 1
        max_gen = max(a.generation for a in agents) or 1

        def fitness_score(agent: AgentRecord) -> float:
            time_score = (agent.total_time_alive / max_time) * 0.4
            offspring_score = (agent.offspring_count / max_offspring) * 0.3
            food_score = (agent.total_food_eaten / max_food) * 0.2
            gen_score = (agent.generation / max_gen) * 0.1
            return time_score + offspring_score + food_score + gen_score

        # Separate favorites and non-favorites
        favorites = [a for a in agents if a.is_favorite]
        non_favorites = [a for a in agents if not a.is_favorite]

        # Sort each group by fitness
        favorites.sort(key=fitness_score, reverse=True)
        non_favorites.sort(key=fitness_score, reverse=True)

        # Favorites first, then fill remaining slots with top performers
        result = favorites[:limit]
        remaining_slots = limit - len(result)
        if remaining_slots > 0:
            result.extend(non_favorites[:remaining_slots])

        return result

    def auto_load_best_agents(self, simulation, count: int = None) -> int:
        """Automatically load the best agents into a simulation.

        Called at simulation startup to continue from previous progress.

        Args:
            simulation: Simulation instance with agents to populate.
            count: Number of agents to load. Defaults to simulation's max_agents.

        Returns:
            Number of agents successfully loaded.
        """
        if count is None:
            count = simulation.config.max_agents

        best_agents = self.get_best_agents(limit=count)
        if not best_agents:
            return 0

        loaded = 0
        agent_wrappers = list(simulation.agents.values())

        for i, record in enumerate(best_agents):
            if i >= len(agent_wrappers):
                break

            wrapper = agent_wrappers[i]
            if self.load_agent_into_wrapper(record.id, wrapper):
                simulation.world.add_entity(wrapper.agent)
                loaded += 1

        return loaded
