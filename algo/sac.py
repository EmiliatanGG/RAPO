import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy


class SACPolicy(nn.Module):
    def __init__(
            self,
            actor,
            critic1,
            critic2,
            risk_critic1,  # 新增
            risk_critic2,  # 新增
            actor_optim,
            critic1_optim,
            critic2_optim,
            risk_critic1_optim,  # 新增
            risk_critic2_optim,  # 新增
            action_space,
            dist,
            tau=0.005,
            gamma=0.99,
            alpha=0.2,
            risk_alpha=0.5,
            risk_critic_lr=3e-4,
            device="cpu"
    ):
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        # 直接使用传入的risk critic网络
        self.risk_critic1, self.risk_critic1_old = risk_critic1, deepcopy(risk_critic1)
        self.risk_critic1_old.eval()
        self.risk_critic2, self.risk_critic2_old = risk_critic2, deepcopy(risk_critic2)
        self.risk_critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim
        self.risk_critic1_optim = risk_critic1_optim  # 使用传入的优化器
        self.risk_critic2_optim = risk_critic2_optim  # 使用传入的优化器

        self.action_space = action_space
        self.dist = dist

        self._tau = tau
        self._gamma = gamma
        self._risk_alpha = risk_alpha

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        
        self.__eps = np.finfo(np.float32).eps.item()

        self._device = device
    
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.risk_critic1.train()  # 新增
        self.risk_critic2.train()  # 新增

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.risk_critic1.eval()  # 新增
        self.risk_critic2.eval()  # 新增
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            # 新增risk网络权重同步
        for o, n in zip(self.risk_critic1_old.parameters(), self.risk_critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.risk_critic2_old.parameters(), self.risk_critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def forward(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def sample_action(self, obs, deterministic=False):
        action, _ = self(obs, deterministic)
        return action.cpu().detach().numpy()

    def learn(self, data):
        obs, actions, next_obs, terminals, rewards, risks = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"], data["risks"]

        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)
        risks = torch.as_tensor(risks).to(self._device)  # 新增risk

        # update critic (原有奖励网络)
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, next_log_probs = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # 新增: update risk critic (风险网络)
        risk_q1, risk_q2 = self.risk_critic1(obs, actions).flatten(), self.risk_critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, _ = self(next_obs)
            next_risk_q = torch.max(  # 注意这里是max，因为risk是惩罚
                self.risk_critic1_old(next_obs, next_actions), self.risk_critic2_old(next_obs, next_actions)
            )
            target_risk_q = risks.flatten() + self._gamma * (1 - terminals.flatten()) * next_risk_q.flatten()
        risk_critic1_loss = ((risk_q1 - target_risk_q).pow(2)).mean()
        self.risk_critic1_optim.zero_grad()
        risk_critic1_loss.backward()
        self.risk_critic1_optim.step()
        risk_critic2_loss = ((risk_q2 - target_risk_q).pow(2)).mean()
        self.risk_critic2_optim.zero_grad()
        risk_critic2_loss.backward()
        self.risk_critic2_optim.step()

        # update actor
        a, log_probs = self(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        risk_q1a, risk_q2a = self.risk_critic1(obs, a).flatten(), self.risk_critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a) +
                      self._risk_alpha * torch.max(risk_q1a, risk_q2a)).mean()  # 注意risk是max
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()


        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/risk_critic1": risk_critic1_loss.item(),  # 新增
            "loss/risk_critic2": risk_critic2_loss.item(),  # 新增
            "misc/risk_alpha": self._risk_alpha,  # 新增
            "misc/mean_risk": risks.mean().item()  # 新增
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result

