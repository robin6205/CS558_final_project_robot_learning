import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        # Set the seed
        try:
            self.env.seed(seed)
        except:
            pass  # If the environment doesn't have a seed method, ignore

        # Env for evaluation.
        self.env_test = env_test
        # Set the seed
        try:
            self.env_test.seed(2**31-seed)
        except:
            pass  # If the environment doesn't have a seed method, ignore

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.seed = seed

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate()
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self):
        episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state, _ = self.env_test.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            
            while not done:
                action = self.algo.exploit(state)
                next_state, reward, terminated, truncated, info = self.env_test.step(action)
                done = terminated or truncated
                
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            episodes += 1
            total_return += episode_return

            if episodes >= self.num_eval_episodes:
                break

        # Log evaluation results
        mean_return = total_return / episodes
        mean_num_steps = num_steps / episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.algo.save_models(os.path.join(self.log_dir, 'best'))
            self.eval_steps += 1

        # Log eval results to console
        print(f'Evaluation: {self.eval_steps}   Mean Return: {mean_return:.2f}   Mean Num Steps: {mean_num_steps:.2f}')
        self.writer.add_scalar('evaluate/mean_return', mean_return, self.steps)
        self.writer.add_scalar('evaluate/mean_num_steps', mean_num_steps, self.steps)

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
