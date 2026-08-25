from dataclasses import dataclass

import numpy as np


@dataclass
class SearchSpace:
    low: np.ndarray
    high: np.ndarray


class AdaptivePSO:
    def __init__(
        self,
        objective,
        space,
        particles=12,
        iterations=20,
        seed=42,
    ):
        self.objective = objective
        self.space = space
        self.particles = particles
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)

    def run(self):
        dimensions = len(self.space.low)
        positions = self.rng.uniform(
            self.space.low,
            self.space.high,
            size=(self.particles, dimensions),
        )
        velocities = np.zeros_like(positions)

        personal_best = positions.copy()
        personal_scores = np.array([
            self.objective(position)
            for position in positions
        ])

        best_index = np.argmin(personal_scores)
        global_best = personal_best[best_index].copy()
        global_score = personal_scores[best_index]

        for iteration in range(self.iterations):
            progress = iteration / max(self.iterations - 1, 1)

            # Exploration decreases and exploitation increases over time.
            inertia = 0.9 - 0.5 * progress
            cognitive = 2.5 - 1.0 * progress
            social = 1.0 + 1.5 * progress

            r1 = self.rng.random(positions.shape)
            r2 = self.rng.random(positions.shape)

            velocities = (
                inertia * velocities
                + cognitive * r1 * (personal_best - positions)
                + social * r2 * (global_best - positions)
            )
            positions = np.clip(
                positions + velocities,
                self.space.low,
                self.space.high,
            )

            for index, position in enumerate(positions):
                score = self.objective(position)

                if score < personal_scores[index]:
                    personal_scores[index] = score
                    personal_best[index] = position.copy()

                if score < global_score:
                    global_score = score
                    global_best = position.copy()

            print(
                f"Iteration {iteration + 1:02d}: "
                f"best validation loss={global_score:.6f}"
            )

        return global_best, global_score
