import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.ppo_crit = utils.PPORewardCriterion() ## Added this for ppo 9/sep/2019
        self.old_sample_logprobs = torch.zeros(50,16).to('cuda')

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices, sc_flag, ppo_flag, clipped_lambda, sc_lambda): ## Added ppo_flag and old_model for ppo 9/sep/2019
        out = {}
        ################ ADDED THIS SECTION for ppo 8/sep/2019
        if ppo_flag:
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max': 0},mode='sample')
            #print("######### SAMPLE LOGPROB#######",sample_logprobs.shape,sample_logprobs) ## REMOVE LATER
            #if self.old_sample_logprobs == None: ## Added this to control the intial null problem of the old policy
            #    self.old_sample_logprobs = sample_logprobs.clone() ## Added this on 11/Sep/2019

            #print('gen_result length:\n',gen_result)
            gts = [gts[_] for _ in gt_indices.tolist()]
            #reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, self.opt)
            #print("Reward given:", reward, len(reward))
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            # loss = self.ppo_crit(sample_logprobs, self.old_sample_logprobs, gen_result.data,reward)  # The loss is the main part, the core of reinforce, I guess, coming from utils.RewardCriterion()
            ###### Added in 24/sep/2019 as a way of combining PPO-clip and scst#######
            loss_ppo = self.ppo_crit(sample_logprobs, self.old_sample_logprobs, gen_result.data,reward)  # The loss is the main part, the core of reinforce, I guess, coming from utils.RewardCriterion()
            loss_sc = self.rl_crit(sample_logprobs, gen_result.data, reward)
            self.old_sample_logprobs = sample_logprobs.clone()
            print("Using sc_lambda: {}\tclipped_lambda: {}".format(sc_lambda,clipped_lambda))
            loss = sc_lambda * loss_sc + clipped_lambda * loss_ppo
            #loss = sc_lambda * loss_sc + clipped_lambda * 1 #********* Replacing with a dummy value c = 1 - 13/oct/2019
            #loss = loss_ppo ## Activate for only Clipped-SC loss
            #########################################################################
            out['reward'] = reward[:, 0].mean()
        else: ##############################################
            if not sc_flag:
                loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
            else:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, self.opt)
                reward = torch.from_numpy(reward).float().to(gen_result.device)
                loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
                out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
