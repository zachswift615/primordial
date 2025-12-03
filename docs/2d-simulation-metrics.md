Core Performance Metrics

1. Survival Time

Most fundamental metric—does learning help agents live longer?

# Track per agent
survival_time = death_time - birth_time

# Compare populations
untrained_mean = mean(survival_times_untrained)
trained_mean = mean(survival_times_trained)
improvement_factor = trained_mean / untrained_mean

# Report: "Trained agents survive 5.2x longer (mean: 1247s vs 240s, p<0.001)"

Variation: Survival curve plotting (like Kaplan-Meier in medical research)

2. Energy Management

How efficiently do they maintain energy?

# Average energy over lifetime
mean_energy = integrate(energy_over_time) / survival_time

# Energy variance (stable vs yo-yo-ing)
energy_stability = std(energy_samples)

# Time spent in danger zone (<20% energy)
critical_time_fraction = sum(energy < 0.2) / total_time

3. Food Acquisition Rate

# Foods eaten per unit time
food_efficiency = total_foods_eaten / survival_time

# Search efficiency: time between spotting food and eating it
mean_acquisition_time = mean(eat_time - first_visible_time)

# Opportunity conversion: % of visible food actually eaten
conversion_rate = foods_eaten / foods_visible_in_lifetime

4. Predator Avoidance

# Damage taken per predator encounter
damage_per_encounter = total_damage / num_encounters

# Escape success rate
escape_rate = successful_escapes / total_encounters

# Reaction time: time from predator visible to flee action
reaction_time = mean(flee_start - predator_visible)

# Safe distance maintenance
mean_distance_from_predators = mean(distances_when_visible)

5. Learning Curves

Plot metrics over time to show improvement:

# Bucketed by experience
for epoch in [0-100, 100-500, 500-1000, 1000+]:
    survival_time[epoch]
    food_efficiency[epoch]
    damage_rate[epoch]

# Report: "Food efficiency improves 3.2x from epoch 0-100 (0.3 foods/min) 
#          to epoch 1000+ (0.96 foods/min)"

Behavioral Emergence Metrics

6. Grouping Behavior

If they "stay in groups":

# Nearest neighbor distance over time
mean_nn_distance = mean(distance_to_nearest_agent)

# Time spent in groups (within threshold distance of others)
group_time_fraction = sum(nn_distance < threshold) / total_time

# Compare: random agents vs trained
random_grouping = 0.15  # baseline from random walk
trained_grouping = 0.67  # 4.5x more grouping

7. Exploration vs Exploitation

# Spatial coverage (how much of map explored)
visited_tiles = unique(positions_visited)
coverage = visited_tiles / total_tiles

# Return rate to known food locations
revisit_rate = visits_to_known_food / total_food_searches

# Balance metric: entropy of position distribution
exploration_entropy = -sum(p * log(p) for p in position_histogram)

8. Decision Quality

# Action diversity (not stuck in loops)
action_entropy = -sum(p * log(p) for p in action_distribution)

# Context-appropriate actions
# When energy < 30%: % time spent moving toward food
food_seeking_when_hungry = moving_toward_food / time_hungry

# When predator nearby: % time spent fleeing vs attacking
flee_rate_in_danger = flee_actions / (flee_actions + other_actions)

Statistical Comparison Framework

Before/After Learning

# Run 100 agents for 50 episodes each
# First 10 episodes vs Last 10 episodes

early_performance = metrics(episodes_0_to_10)
late_performance = metrics(episodes_40_to_50)

improvement = (late - early) / early
t_test(early, late)  # Statistical significance

Ablation Studies

# Full model vs variants
baseline = random_actions
perception_only = trained_encoder + random_actions
reward_only = random_encoder + trained_reward_head
full_model = trained_all

# Report which components matter most

Difficulty Scaling

# Does learning transfer to harder environments?
easy = {predator_speed: 0.5, food_scarcity: 0.1}
medium = {predator_speed: 1.0, food_scarcity: 0.3}
hard = {predator_speed: 1.5, food_scarcity: 0.5}

# Trained agents maintain performance advantage across difficulties?
performance_gap = trained_survival / untrained_survival
# Easy: 5.2x, Medium: 3.8x, Hard: 2.1x (learning helps even in hard mode)

Visualization for Papers/Demos

Figure 1: Survival Curves
Survival probability over time
100% |    Trained ────────────╮
    |                        │╲
50% |                        │ ╲___
    |    Untrained ──────╮   │
    |                    ╲___│
0% |________________________╲
    0     500    1000   1500   2000 steps

Figure 2: Learning Progression
Food Efficiency (foods/minute)
1.0 |                    ╱─────
    |                 ╱─╯
0.5 |           ╱────╯
    |     ╱────╯
0.0 |────╯___________________
    0   200  400  600  800  1000 episodes

Figure 3: Spatial Heatmaps
Untrained agents:        Trained agents:
■ ■ □ □ □ ■ ■ ■         ■ ■ ■ ■ ■ ■ ■ ■
■ ■ □ □ □ □ □ □         ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □         ■ ■ ■ ■ ■ ■ ■ ■
(Random wandering)       (Systematic coverage)

Concrete Implementation

class SimulationMetrics:
    def __init__(self):
        self.agent_lifespans = []
        self.foods_eaten = []
        self.damage_taken = []
        self.positions = []

    def record_death(self, agent):
        self.agent_lifespans.append(agent.survival_time)
        self.foods_eaten.append(agent.foods_eaten)
        self.damage_taken.append(agent.total_damage)

    def summary(self):
        return {
            'mean_survival': np.mean(self.agent_lifespans),
            'food_efficiency': np.mean(self.foods_eaten) / np.mean(self.agent_lifespans),
            'survival_95th': np.percentile(self.agent_lifespans, 95),
            'damage_rate': np.mean(self.damage_taken) / np.mean(self.agent_lifespans)
        }

    def compare(self, other):
        """Statistical comparison with another population"""
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(self.agent_lifespans, other.agent_lifespans)
        improvement = np.mean(self.agent_lifespans) / np.mean(other.agent_lifespans)
        return f"{improvement:.2f}x improvement (p={p_value:.4f})"

The Key Result

For your paper/README, you want to show:

"Trained agents survive 5.2x longer than untrained (mean: 1247s vs 240s, t=12.4, p<0.001), achieve 3.8x higher food efficiency (0.96 vs 0.25 foods/min), and take 2.3x less damage per 
predator encounter (15 vs 35 health). Learning curves show continued improvement over 1000 episodes with no catastrophic forgetting."

That's quantitative evidence of what you're seeing qualitatively. Which metrics resonate most with what you're observing?