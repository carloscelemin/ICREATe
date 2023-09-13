import gym
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import configparser
from model import PolicyEnsemble, PolicyRAMEnsemble
from keyboard import Keyboard
import time
import numpy as np
import cv2
import logging

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def plot(u_e,u_a): # if uncertainty is high plot figure
    if u_e > config["AGENT"].getfloat("th_ue") or u_a > config["AGENT"].getfloat("th_ua"):
        cv2.imshow('FB', image)
    else:
        cv2.imshow('FB', dark_im)
    cv2.waitKey(1)

def parseFeedback(h,states,actions,h_signals,state,action,state_1,action_1,count_T_g,u_e,u_a):
    if h >= 0:  # CC
        action = h
        states = np.append(states, state.reshape((1, -1)), axis=0)
        actions.append(action)
        h_signals.append(1)
    if h == -20:  # punishment
        states = np.append(states, state_1.reshape((1, -1)), axis=0)
        actions.append(action_1)
        h_signals.append(-1)
    if h == -10:  # reward
        states = np.append(states, state_1.reshape((1, -1)), axis=0)
        actions.append(action_1)
        h_signals.append(1)

    if h == -1 and (u_e>config["AGENT"].getfloat("th_ue") or u_a>config["AGENT"].getfloat("th_ua") ) and config['AGENT'].getboolean('passive_rewarding'):
        count_T_g += config["GENERAL"].getfloat("render_delay")
        if count_T_g > 1.3: # 1.3 seconds of reaction of the teacher
            states = np.append(states, state_1.reshape((1, -1)), axis=0)
            actions.append(action_1)
            h_signals.append(1)
            print(f"passive feedback")
    else:
        count_T_g = 0

    return states, actions, h_signals, action, count_T_g

def training(policy,states,actions,h_signals,loss,loss_m,optimizer):
    if len(actions)<config['AGENT'].getint('batch_size')*config['AGENT'].getint('min_number_of_batches'):
        return
    x_tr = torch.tensor(states, dtype=torch.float)
    ylabels = np.zeros((len(actions), config['GENERAL'].getint('output_size')))
    ylabels[np.arange(len(actions)), actions] = 1
    ylabels[h_signals == -1, :] = -1 * (ylabels[h_signals == -1, :] - 1) # creates one cold vector for punished samples
    y_tr = torch.tensor(ylabels, dtype=torch.float)
    policyDataset = torch.utils.data.TensorDataset(x_tr,y_tr)
    policyLoader = torch.utils.data.DataLoader(policyDataset, batch_size=64, shuffle=True)
    # policy training
    policy.train()
    for epoch in range(config['AGENT'].getint('epochs')):
        for _, (x,y) in enumerate(policyLoader):
            head = np.random.randint(5) # choosing randomly a head
            x, y = x.to(device), y.to(device)
            y_pred = policy(x)
            optimizer.zero_grad()
            loss_head = loss(y_pred[head],y)
            loss_head.backward()
            optimizer.step()

        if np.random.rand() < 0.1:
            statesNonNegFB = states[h_signals != -1] # considering only demonstrated instances or with positive FB
            actionsNonNegFB = actions[h_signals != -1]
            statesNegFB = states[h_signals == -1] # considering instances that were punished
            actionsNegFB = actions[h_signals == -1]
            if len(statesNonNegFB) > 0:
                actionsNonNegFB_pred,_,_ = policy.predict(torch.tensor(statesNonNegFB, dtype=torch.float))
                mistakes1 = np.zeros((len(actionsNonNegFB),1))
                mistakes1[(actionsNonNegFB - actionsNonNegFB_pred.detach().cpu().numpy()) != 0, 0] = 1
                mistakes = mistakes1
                statesM = statesNonNegFB

            if len(statesNegFB) > 0:
                actionsNegFB_pred,_,_ = policy.predict(torch.tensor(statesNegFB, dtype=torch.float))
                mistakes2 = np.zeros((len(actionsNegFB), 1))
                mistakes2[(actionsNegFB - actionsNegFB_pred.detach().cpu().numpy()) != 0, 0] = 1
                mistakes = np.concatenate((mistakes,mistakes2),axis=0) if (len(statesNonNegFB) > 0) else mistakes2
                statesM = np.concatenate((statesM,statesNegFB),axis=0) if (len(statesNonNegFB) > 0) else statesNegFB

            x_m_tr = torch.tensor(statesM, dtype=torch.float)
            y_m_tr = torch.tensor(mistakes, dtype=torch.float)
            mistakesDataset = torch.utils.data.TensorDataset(x_m_tr, y_m_tr)
            mistakesLoader = torch.utils.data.DataLoader(mistakesDataset, batch_size=64, shuffle=True)
            for _, (xm, ym) in enumerate(mistakesLoader):
                xm, ym = xm.to(device), ym.to(device)
                ym_pred = policy(xm)
                optimizer.zero_grad()
                loss_ms = loss_m(ym_pred[-1], ym)
                loss_ms.backward()
                optimizer.step()
    #logger.info(f'the mistakes {len(mistakes)} and sum is {np.sum(mistakes)}')
    #logger.info(mistakes)

    return

def main():
    env = gym.make(config['GENERAL']['environment'])  # create environment
    if config['GENERAL']['render']:
        env.render()
    human_feedback = Keyboard(env, config)
    policy = models[config['AGENT']['architecture']](config).to(device)
    loss = nn.MSELoss()
    loss_m = nn.BCELoss()
    optimizer = optim.Adam(policy.parameters(), lr=config['AGENT'].getfloat('learning_rate'))
    states = np.empty(shape=[0,config['GENERAL'].getint('input_size')])
    actions = []
    h_signals = []

    for episode in range(config['GENERAL'].getint('max_episode_count')):
        done = False
        policy.eval()
        step_count = 0
        state = env.reset()
        state_1 = state
        action_1,u_e,u_a = policy.predict(torch.unsqueeze(torch.tensor(state, dtype=torch.float),0))
        count_T_g = 0 # timer for passive rewarding
        rewards = 0
        while not done:
            env.render()  # CC
            if config['GENERAL'].getboolean('render_delay_flag'):  # CC
                plot(u_e, u_a)
                time.sleep(config['GENERAL'].getfloat('render_delay'))  # CC
            h = human_feedback.get_h()  # CC Get feedback signal
            action, u_e, u_a = policy.predict(torch.unsqueeze(torch.tensor(state, dtype=torch.float),0))
            #print(f'U_e: {u_e}    U_a: {u_a}')
            #print(policy(torch.unsqueeze(torch.tensor(state, dtype=torch.float),0)))
            action, u_e, u_a = action.item(), u_e.item(), u_a.item()
            states, actions, h_signals, action, count_T_g = parseFeedback(h, states, actions, h_signals, state, action, state_1, action_1, count_T_g, u_e, u_a)
            # Get new state and reward from environment
            state_1, action_1 = state, action
            state, reward, done, _ = env.step(action)
            rewards += reward
        logger.info(f"Episode: {episode}    return: {rewards}    feedback: {len(actions)}")
        training(policy, np.array(states), np.array(actions), np.array(h_signals), loss, loss_m, optimizer)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='skiing_ICREATe', help='select file in config_files folder')
    args = parser.parse_args()
    config_file = args.config_file
    config = configparser.ConfigParser()
    config.read('config_files/' + config_file + '.ini')
    image = cv2.imread('config_files/notsure1.jpeg')
    dark_im = image*0
    plot( 0, 0)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    models = {'Simple':PolicyEnsemble,'RAMmodel':PolicyRAMEnsemble}
    main()
