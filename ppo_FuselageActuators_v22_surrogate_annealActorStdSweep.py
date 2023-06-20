# Author: Tim Lutz
# Based on docs and code found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from AssemblyGym.envs import FuselageActuators 


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Fuselage Actuators",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="tlutz",
        help="the entity (team) of wandb's project")
    parser.add_argument("--resume", type=any, default=False,
        help="resume from a previous run")
    parser.add_argument("--resume-run-id", type=str, default=None,
        help="resume from a previous run")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="FuselageActuators-v22",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=16384000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=12,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--actor-logstd", type=float, default=-5,
        help="exponent for actor_std initialization")

    # Environment specific arguments
    parser.add_argument("--n_actions", type=int, default=10,
        help="the maximum number of non-zero actions")
    parser.add_argument("--mode", type=str, default="Surrogate",
        help="instantiate environment for training, testing, benchmark, or surrogate")
    parser.add_argument("--record", type=bool, default=False,
        help="record interaction with environment")
    parser.add_argument("--norm-reward", type=bool, default=False,
        help="enable wrapper to normalize reward")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, n_actions, mode, record):
    def thunk():
        env = gym.make(env_id, n_actuators=n_actions, mode=mode, record=record, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)#, new_step_api=False)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if args.norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # layers for self.actor_mean
        self.fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        
        self.actor_logstd = nn.Parameter(args.actor_logstd*torch.ones(1, np.prod(envs.single_action_space.shape)), requires_grad=False)  # initial action_std = exp(actor_logstd)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None, scaleStd=1):
        # Start with standard MLP
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        action_mean = torch.tanh(self.fc3(x))
        # Build action distribution
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)*scaleStd
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if action == "deterministic":
            action = action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

# os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    args = parse_args()
    for n_act in [1,3,5,6,7,8,9,10,12,14,16,18]:
        args.n_actions = n_act
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.require("service")

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                resume=args.resume,
                mode="online",
                config=vars(args),
                name=run_name,
                monitor_gym=False,
                save_code=True
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, n_act, args.mode, args.record) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = int(args.total_timesteps // args.batch_size)

        CHECKPOINT_FREQUENCY = 1024
        starting_update = 1

        for update in range(starting_update, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs, scaleStd=frac)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if step % args.num_steps == 0: 
                    # Log the reward
                    print(f"global_step={global_step}, episodic_return={reward}")
                    writer.add_scalar("charts/episodic_return", np.mean(reward), global_step)
                    writer.add_scalar("charts/initialError", np.mean(info["initError"]), global_step)
                    writer.add_scalar("charts/finalError", np.mean(info["Error"]), global_step)
                    writer.add_scalar("charts/initMaxDev", np.mean(info["initMaxDev"]), global_step)
                    writer.add_scalar("charts/maxDev", np.mean(info["maxDev"]), global_step)

                    if args.track:
                        wandb.log({"episodic_return": reward,"global_step": global_step}, step=global_step)
                        wandb.log({"episodic_return": np.mean(reward),"initial_error": np.mean(info["initError"]), "final_error": np.mean(info["Error"]), "init_max_dev":np.mean(info["initMaxDev"]) ,"max_dev":np.mean(info["maxDev"]), "global_step": global_step}, step=global_step)


            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], scaleStd=frac)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/update", update, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # TB doesn't show on wandb web interface on Windows, so log directly to 
            if args.track:
                wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)
                wandb.log({"update": update}, step=global_step)
                wandb.log({"SPS": int(global_step / (time.time() - start_time))}, step=global_step)
                wandb.log({"losses/value_loss": v_loss.item()}, step=global_step)
                wandb.log({"losses/policy_loss": pg_loss.item()}, step=global_step)
                wandb.log({"losses/entropy": entropy_loss.item()}, step=global_step)
                wandb.log({"losses/old_approx_kl": old_approx_kl.item()}, step=global_step)
                wandb.log({"losses/approx_kl": approx_kl.item()}, step=global_step)
                wandb.log({"losses/clipfrac": np.mean(clipfracs)}, step=global_step)
                wandb.log({"losses/explained_variance": explained_var}, step=global_step)

            if args.track:
                # make sure to tune `CHECKPOINT_FREQUENCY` 
                # so models are not saved too frequently
                if update % CHECKPOINT_FREQUENCY == 0 or update == num_updates:
                    torch.save(agent.state_dict(), f"{wandb.run.dir}/agent_{global_step}steps.pt")
                    wandb.save(f"{wandb.run.dir}/agent_{global_step}steps.pt", policy="now")
            elif update % CHECKPOINT_FREQUENCY == 0 or update == num_updates:
                torch.save(agent.state_dict(), f"./Trained agents/{run_name}/agent_{global_step}steps.pt")

        envs.close()


        # Create evaluation env
        # envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.n_actions, "Test", args.record)

        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, n_act, "Test", args.record) for i in range(1)]
        )
        # envs = gym.make(args.env_id, n_actuators=args.n_actions, mode="Test", record=False, seed=args.seed)

        # Log summaries for evaluation metrics
        wandb.define_metric("eval/final_error", summary="mean")
        wandb.define_metric("eval/max_dev", summary="mean")
        wandb.define_metric("eval/max_force", summary="mean")
        wandb.define_metric("eval/final_error", summary="max")
        wandb.define_metric("eval/max_dev", summary="max")
        wandb.define_metric("eval/max_force", summary="max")

        for i in range(100):
            next_obs = torch.Tensor(envs.reset()).to(device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action="deterministic")
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            if args.track:
                wandb.log({"eval/reward": np.mean(reward),"eval/initial_error": np.mean(info["initError"]), "eval/final_error": np.mean(info["Error"]), "eval/init_max_dev":np.mean(info["initMaxDev"]) ,"eval/max_dev":np.mean(info["maxDev"]), "eval/max_force":np.max(info["Forces"]), "eval/step": i})

        writer.close()
        wandb.finish()
        envs.close()