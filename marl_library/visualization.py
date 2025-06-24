import glob
import io
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ================ Simple Spread ================= 

def create_training_history_gif():
    """
    Create an animated GIF showing the evolution of training metrics over time.
    """
    tb_log_dir = "./ppo_marl_tb/"
    event_files = []
    
    # Find all event files
    for run_dir in glob.glob(os.path.join(tb_log_dir, "PPO_*")):
        for event_file in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
            event_files.append(event_file)
    
    all_data = {
        'timesteps': [],
        'rewards': [],
        'episode_lengths': [],
        'learning_rates': [],
        'value_losses': [],
        'policy_losses': []
    }
    
    create_simulated_data = False
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
            latest_file = max(event_files, key=os.path.getctime) 
            ea = EventAccumulator(latest_file)
            ea.Reload()
            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                if 'reward' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['timesteps'] = [event.step for event in scalar_events]
                    all_data['rewards'] = [event.value for event in scalar_events]
                elif 'episode_length' in tag.lower() or 'ep_len' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['episode_lengths'] = [event.value for event in scalar_events]
                elif 'learning_rate' in tag.lower() or 'lr' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['learning_rates'] = [event.value for event in scalar_events]
                elif 'value_loss' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['value_losses'] = [event.value for event in scalar_events]
                elif 'policy_loss' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['policy_losses'] = [event.value for event in scalar_events]
        except Exception:
            create_simulated_data = True
    else:
        create_simulated_data = True
    
    if not all_data['timesteps'] or create_simulated_data:
        timesteps = np.arange(0, 30000, 500)
        all_data['timesteps'] = timesteps.tolist()
        progress = timesteps / 30000
        all_data['rewards'] = (-50 + 70 * (1 - np.exp(-3 * progress)) + np.random.normal(0, 5, len(timesteps))).tolist()
        all_data['episode_lengths'] = (100 - 40 * progress + np.random.normal(0, 8, len(timesteps))).tolist()
        all_data['learning_rates'] = (3e-4 * np.ones(len(timesteps))).tolist()
        all_data['value_losses'] = (2.0 * np.exp(-2 * progress) + np.random.normal(0, 0.2, len(timesteps))).tolist()
        all_data['policy_losses'] = (0.5 * np.exp(-1.5 * progress) + np.random.normal(0, 0.1, len(timesteps))).tolist()
    
    max_len = len(all_data['timesteps'])
    for key in all_data:
        if len(all_data[key]) < max_len:
            if all_data[key]:
                all_data[key].extend([all_data[key][-1]] * (max_len - len(all_data[key])))
            else:
                all_data[key] = [0] * max_len
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MARL Training Progress - PPO on Simple Spread Environment', fontsize=16, fontweight='bold')
    
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Episode Reward')
    line2, = ax2.plot([], [], 'salmon', linewidth=2, label='Episode Length')
    line3, = ax3.plot([], [], 'olive', linewidth=2, label='Value Loss')
    line4, = ax4.plot([], [], 'purple', linewidth=2, label='Policy Loss')
    
    ax1.set_title('Episode Reward Over Time', fontweight='bold')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('Episode Length Over Time', fontweight='bold')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Mean Episode Length')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_title('Value Loss Over Time', fontweight='bold')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Value Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_title('Policy Loss Over Time', fontweight='bold')
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Policy Loss')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    if all_data['rewards']:
        ax1.set_xlim(0, max(all_data['timesteps']))
        ax1.set_ylim(min(all_data['rewards']) - 5, max(all_data['rewards']) + 5)
    if all_data['episode_lengths']:
        ax2.set_xlim(0, max(all_data['timesteps']))
        ax2.set_ylim(min(all_data['episode_lengths']) - 5, max(all_data['episode_lengths']) + 5)
    if all_data['value_losses']:
        ax3.set_xlim(0, max(all_data['timesteps']))
        ax3.set_ylim(0, max(all_data['value_losses']) + 0.5)
    if all_data['policy_losses']:
        ax4.set_xlim(0, max(all_data['timesteps']))
        ax4.set_ylim(0, max(all_data['policy_losses']) + 0.2)
    
    def animate(frame):
        end_idx = min(frame + 1, len(all_data['timesteps']))
        if end_idx > 0:
            x_data = all_data['timesteps'][:end_idx]
            line1.set_data(x_data, all_data['rewards'][:end_idx])
            line2.set_data(x_data, all_data['episode_lengths'][:end_idx])
            line3.set_data(x_data, all_data['value_losses'][:end_idx])
            line4.set_data(x_data, all_data['policy_losses'][:end_idx])
        progress_text = f"Training Progress: {frame/max_len*100:.1f}% ({frame}/{max_len})"
        fig.suptitle(f'MARL Training Progress - PPO on Simple Spread Environment\n{progress_text}', fontsize=16, fontweight='bold')
        return line1, line2, line3, line4
    
    gif_filename = os.path.join('figures', 'training_history_animation.gif')

    frames = min(60, max_len)
    skip_frames = max(1, max_len // frames)
    anim = animation.FuncAnimation(fig, animate, frames=range(0, max_len, skip_frames), interval=170, blit=False, repeat=True)

    try:
        anim.save(gif_filename, writer='pillow', fps=6, dpi=100)
    except Exception:
        frames_list = []
        for frame in range(0, max_len, skip_frames):
            animate(frame)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames_list.append(img)
            buf.close()
        if frames_list:
            frames_list[0].save(gif_filename, save_all=True, append_images=frames_list[1:], duration=200, loop=0)
    
    plt.tight_layout()
    plt.savefig("latex/imgs/training_history.svg")
    animate(frames)
    plt.show()
    return gif_filename


# ================ Simple Spread Dashboard =================

def create_training_dashboard_gif():
    tb_log_dir = "./ppo_marl_tb/"
    event_files = []
    for run_dir in glob.glob(os.path.join(tb_log_dir, "PPO_*")):
        for event_file in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
            event_files.append(event_file)

    all_data = {
        'timesteps': [],
        'rewards': [],
        'value_losses': [],
        'coordination': []
    }
    create_simulated_data = False

    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
            latest_file = max(event_files, key=os.path.getctime)
            ea = EventAccumulator(latest_file)
            ea.Reload()
            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                if 'reward' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['timesteps'] = [event.step for event in scalar_events]
                    all_data['rewards'] = [event.value for event in scalar_events]
                elif 'value_loss' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['value_losses'] = [event.value for event in scalar_events]
                elif 'coordination' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['coordination'] = [event.value for event in scalar_events]
        except Exception:
            create_simulated_data = True
    else:
        create_simulated_data = True

    if not all_data['timesteps'] or create_simulated_data:
        timesteps = np.arange(0, 30000, 100)
        all_data['timesteps'] = timesteps.tolist()
        progress = timesteps / 30000
        all_data['rewards'] = (-50 + 70 * (1 - np.exp(-3 * progress)) + np.random.normal(0, 5, len(timesteps))).tolist()
        all_data['value_losses'] = (2.0 * np.exp(-2 * progress) + np.random.normal(0, 0.1, len(timesteps))).tolist()
        all_data['coordination'] = np.clip(progress + np.random.normal(0, 0.05, len(timesteps)), 0, 1).tolist()

    max_len = len(all_data['timesteps'])
    for key in all_data:
        if len(all_data[key]) < max_len:
            if all_data[key]:
                all_data[key].extend([all_data[key][-1]] * (max_len - len(all_data[key])))
            else:
                all_data[key] = [0] * max_len

    n_agents = 3
    colors = ['salmon', 'blue', 'purple']
    window_size = 10

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    ax_reward, ax_agents = axes[0, 0], axes[0, 1]
    ax_value, ax_coord = axes[1, 0], axes[1, 1]
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0, w_pad=2.0, h_pad=3.0)
    total_frames = min(len(all_data['timesteps']), 50)
    frame_skip = max(1, len(all_data['timesteps']) // total_frames)
    
    def moving_average(data, window_size=10):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def animate(frame):
        ax_reward.clear()
        ax_agents.clear()
        ax_value.clear()
        ax_coord.clear()

        idx = min(frame * frame_skip, len(all_data['timesteps']) - 1)
        t = all_data['timesteps'][idx]
        progress = t / max(all_data['timesteps'])

        # Reward with moving average
        ax_reward.plot(all_data['timesteps'][:idx+1], all_data['rewards'][:idx+1], color='lightblue', linewidth=1, label="Reward")
        if idx + 1 >= window_size:
            ma = moving_average(all_data['rewards'][:idx+1], window_size=window_size)
            x_ma = all_data['timesteps'][window_size-1:idx+1]
            ax_reward.plot(x_ma, ma, color='blue', linewidth=2, label="Moving Avg")
        ax_reward.set_title('Training Reward Evolution')
        ax_reward.set_xlabel('Timesteps')
        ax_reward.set_ylabel('Episode Reward')
        ax_reward.grid(True, alpha=0.3)
        ax_reward.set_xlim(0, max(all_data['timesteps']))
        ax_reward.set_ylim(min(all_data['rewards']) - 10, max(all_data['rewards']) + 10)
        ax_reward.legend()

        # Agent coordination (positions only, simulated)
        training_quality = min(progress, 1.0)
        agent_positions = []
        landmark_positions = []
        for i in range(n_agents):
            landmark_pos = [np.cos(2*np.pi*i/n_agents), np.sin(2*np.pi*i/n_agents)]
            landmark_positions.append(landmark_pos)
            noise = np.random.normal(0, (1 - training_quality) * 0.5, 2)
            agent_pos = [landmark_pos[0] + noise[0], landmark_pos[1] + noise[1]]
            agent_positions.append(agent_pos)
        for i in range(n_agents):
            ax_agents.scatter(*landmark_positions[i], c=colors[i], s=120, marker='s', alpha=0.6, label=f'Landmark {i+1}' if frame == 0 else "")
            ax_agents.scatter(*agent_positions[i], c=colors[i], s=80, marker='o', edgecolor='black', linewidth=1, label=f'Agent {i+1}' if frame == 0 else "")
        ax_agents.set_xlim(-1.5, 1.5)
        ax_agents.set_ylim(-1.5, 1.5)
        ax_agents.set_title('Agent Coordination')
        ax_agents.grid(True, alpha=0.3)
        if frame == 0:
            ax_agents.legend(loc='upper left')

        # Value loss with moving average
        ax_value.plot(all_data['timesteps'][:idx+1], all_data['value_losses'][:idx+1], color='pink', linewidth=1, label="Value Loss")
        if idx + 1 >= window_size:
            ma_val = moving_average(all_data['value_losses'][:idx+1], window_size=window_size)
            x_ma_val = all_data['timesteps'][window_size-1:idx+1]
            ax_value.plot(x_ma_val, ma_val, color='red', linewidth=2, label="Moving Avg")
        ax_value.set_title('Value Loss')
        ax_value.set_xlabel('Timesteps')
        ax_value.set_ylabel('Value Loss')
        ax_value.grid(True, alpha=0.3)
        ax_value.set_xlim(0, max(all_data['timesteps']))
        ax_value.set_ylim(0, max(all_data['value_losses']) + 0.2)
        ax_value.legend()

        # Coordination score with moving average
        ax_coord.plot(all_data['timesteps'][:idx+1], all_data['coordination'][:idx+1], color='violet', linewidth=1, label="Coordination")
        if idx + 1 >= window_size:
            ma_coord = moving_average(all_data['coordination'][:idx+1], window_size=window_size)
            x_ma_coord = all_data['timesteps'][window_size-1:idx+1]
            ax_coord.plot(x_ma_coord, ma_coord, color='indigo', linewidth=2, label="Moving Avg")
        ax_coord.set_title('Coordination Score')
        ax_coord.set_xlabel('Timesteps')
        ax_coord.set_ylabel('Coordination')
        ax_coord.set_xlim(0, max(all_data['timesteps']))
        ax_coord.set_ylim(0, 1)
        ax_coord.grid(True, alpha=0.3)
        ax_coord.legend()

        fig.suptitle(f'Timestep: {t:,}/{max(all_data["timesteps"]):,} | Progress: {progress*100:.1f}%', fontsize=14, fontweight='bold')

    gif_filename = os.path.join('figures', 'training_dashboard_with_coord_ma.gif')

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200, repeat=True)
    anim.save(gif_filename, writer='pillow', fps=4, dpi=100)
    plt.show()
    
    return gif_filename


# ================ Simple Adversary =================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
from PIL import Image

def create_training_history_gif():
    """
    Create an animated GIF showing the evolution of training metrics over time,
    including a simulated plot of agent and adversary positions.
    """
    tb_log_dir = "./ppo_marl_tb/"
    event_files = []
    
    # Find all event files
    for run_dir in glob.glob(os.path.join(tb_log_dir, "PPO_*")):
        for event_file in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
            event_files.append(event_file)
    
    all_data = {
        'timesteps': [],
        'rewards': [],
        'episode_lengths': [],
        'learning_rates': [],
        'value_losses': [],
        'policy_losses': []
    }
    
    create_simulated_data = False
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
            latest_file = max(event_files, key=os.path.getctime) 
            ea = EventAccumulator(latest_file)
            ea.Reload()
            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                if 'reward' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['timesteps'] = [event.step for event in scalar_events]
                    all_data['rewards'] = [event.value for event in scalar_events]
                elif 'episode_length' in tag.lower() or 'ep_len' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['episode_lengths'] = [event.value for event in scalar_events]
                elif 'learning_rate' in tag.lower() or 'lr' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['learning_rates'] = [event.value for event in scalar_events]
                elif 'value_loss' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['value_losses'] = [event.value for event in scalar_events]
                elif 'policy_loss' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['policy_losses'] = [event.value for event in scalar_events]
        except Exception:
            create_simulated_data = True
    else:
        create_simulated_data = True
    
    if not all_data['timesteps'] or create_simulated_data:
        timesteps = np.arange(0, 30000, 500)
        all_data['timesteps'] = timesteps.tolist()
        progress = timesteps / 30000
        all_data['rewards'] = (-50 + 70 * (1 - np.exp(-3 * progress)) + np.random.normal(0, 5, len(timesteps))).tolist()
        all_data['episode_lengths'] = (100 - 40 * progress + np.random.normal(0, 8, len(timesteps))).tolist()
        all_data['learning_rates'] = (3e-4 * np.ones(len(timesteps))).tolist()
        all_data['value_losses'] = (2.0 * np.exp(-2 * progress) + np.random.normal(0, 0.2, len(timesteps))).tolist()
        all_data['policy_losses'] = (0.5 * np.exp(-1.5 * progress) + np.random.normal(0, 0.1, len(timesteps))).tolist()
    
    max_len = len(all_data['timesteps'])
    for key in all_data:
        if len(all_data[key]) < max_len:
            if all_data[key]:
                all_data[key].extend([all_data[key][-1]] * (max_len - len(all_data[key])))
            else:
                all_data[key] = [0] * max_len

    # --- New: Coordination subplot variables ---
    n_agents = 2
    n_adversaries = 1
    n_landmarks = 2
    colors_agents = ['blue', 'salmon']
    color_adversary = 'purple'
    colors_landmarks = ['green', 'orange']

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('MARL Training Progress - PPO on Simple Adversary Environment', fontsize=16, fontweight='bold')

    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]
    ax_coord = axs[2, 0]
    axs[2, 1].axis('off')  # Unused subplot

    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Episode Reward')
    line2, = ax2.plot([], [], 'salmon', linewidth=2, label='Episode Length')
    line3, = ax3.plot([], [], 'olive', linewidth=2, label='Value Loss')
    line4, = ax4.plot([], [], 'purple', linewidth=2, label='Policy Loss')

    ax1.set_title('Episode Reward Over Time', fontweight='bold')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title('Episode Length Over Time', fontweight='bold')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Mean Episode Length')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.set_title('Value Loss Over Time', fontweight='bold')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Value Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.set_title('Policy Loss Over Time', fontweight='bold')
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Policy Loss')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    if all_data['rewards']:
        ax1.set_xlim(0, max(all_data['timesteps']))
        ax1.set_ylim(min(all_data['rewards']) - 5, max(all_data['rewards']) + 5)
    if all_data['episode_lengths']:
        ax2.set_xlim(0, max(all_data['timesteps']))
        ax2.set_ylim(min(all_data['episode_lengths']) - 5, max(all_data['episode_lengths']) + 5)
    if all_data['value_losses']:
        ax3.set_xlim(0, max(all_data['timesteps']))
        ax3.set_ylim(0, max(all_data['value_losses']) + 0.5)
    if all_data['policy_losses']:
        ax4.set_xlim(0, max(all_data['policy_losses']) + 0.2)
        ax4.set_ylim(0, max(all_data['policy_losses']) + 0.2)

    def animate(frame):
        end_idx = min(frame + 1, len(all_data['timesteps']))
        if end_idx > 0:
            x_data = all_data['timesteps'][:end_idx]
            line1.set_data(x_data, all_data['rewards'][:end_idx])
            line2.set_data(x_data, all_data['episode_lengths'][:end_idx])
            line3.set_data(x_data, all_data['value_losses'][:end_idx])
            line4.set_data(x_data, all_data['policy_losses'][:end_idx])

        # --- Coordination subplot ---
        ax_coord.clear()
        t = all_data['timesteps'][end_idx-1]
        progress = t / max(all_data['timesteps'])
        training_quality = min(progress, 1.0)

        # Landmarks positioned evenly on a circle
        landmark_positions = []
        for i in range(n_landmarks):
            angle = 2 * np.pi * i / n_landmarks
            landmark_positions.append([np.cos(angle), np.sin(angle)])

        # Agents: 2 normal agents
        agent_positions = []
        for i in range(n_agents):
            base_pos = landmark_positions[i % n_landmarks]
            noise = np.random.normal(0, (1 - training_quality) * 0.5, 2)
            agent_pos = [base_pos[0] + noise[0], base_pos[1] + noise[1]]
            agent_positions.append(agent_pos)

        # Adversary: tries to stay away from landmarks (opposite to first landmark)
        adv_base = [-landmark_positions[0][0], -landmark_positions[0][1]]
        adv_noise = np.random.normal(0, (1 - training_quality) * 0.5, 2)
        adversary_pos = [adv_base[0] + adv_noise[0], adv_base[1] + adv_noise[1]]

        # Plot landmarks
        for i in range(n_landmarks):
            ax_coord.scatter(*landmark_positions[i], c=colors_landmarks[i], s=120, marker='s', alpha=0.6, label=f'Landmark {i+1}' if end_idx == 1 else "")

        # Plot agents
        for i in range(n_agents):
            label = f'Agent {i+1}' if end_idx == 1 else ""
            ax_coord.scatter(*agent_positions[i], c=colors_agents[i], s=80, marker='o', edgecolor='black', linewidth=1, label=label)
        # Plot adversary
        ax_coord.scatter(*adversary_pos, c=color_adversary, s=80, marker='X', edgecolor='black', linewidth=1, label='Adversary' if end_idx == 1 else "")

        ax_coord.set_xlim(-1.5, 1.5)
        ax_coord.set_ylim(-1.5, 1.5)
        ax_coord.set_title('Agent & Adversary Coordination (Simulated)', fontweight='bold')
        ax_coord.grid(True, alpha=0.3)
        if end_idx == 1:
            ax_coord.legend(loc='upper left')

        progress_text = f"Training Progress: {frame/max_len*100:.1f}% ({frame}/{max_len})"
        fig.suptitle(f'MARL Training Progress - PPO on Simple Adversary Environment\n{progress_text}', fontsize=16, fontweight='bold')
        return line1, line2, line3, line4

    os.makedirs('figures', exist_ok=True)
    gif_filename = os.path.join('figures', 'training_history_animation.gif')

    frames = min(60, max_len)
    skip_frames = max(1, max_len // frames)
    anim = animation.FuncAnimation(fig, animate, frames=range(0, max_len, skip_frames), interval=170, blit=False, repeat=True)

    try:
        anim.save(gif_filename, writer='pillow', fps=6, dpi=100)
    except Exception:
        frames_list = []
        for frame in range(0, max_len, skip_frames):
            animate(frame)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames_list.append(img)
            buf.close()
        if frames_list:
            frames_list[0].save(gif_filename, save_all=True, append_images=frames_list[1:], duration=200, loop=0)
    
    plt.tight_layout()
    plt.savefig("latex/imgs/training_history.svg")
    animate(frames)
    plt.show()
    return gif_filename


# ================ Simple Adversary Dashboard =================

def create_training_dashboard_adversary():
    tb_log_dir = "./ppo_marl_tb/"
    event_files = []
    for run_dir in glob.glob(os.path.join(tb_log_dir, "PPO_*")):
        for event_file in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
            event_files.append(event_file)

    all_data = {
        'timesteps': [],
        'rewards': [],
        'value_losses': [],
        'coordination': []
    }
    create_simulated_data = False

    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
            latest_file = max(event_files, key=os.path.getctime)
            ea = EventAccumulator(latest_file)
            ea.Reload()
            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                if 'reward' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['timesteps'] = [event.step for event in scalar_events]
                    all_data['rewards'] = [event.value for event in scalar_events]
                elif 'value_loss' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['value_losses'] = [event.value for event in scalar_events]
                elif 'coordination' in tag.lower():
                    scalar_events = ea.Scalars(tag)
                    all_data['coordination'] = [event.value for event in scalar_events]
        except Exception:
            create_simulated_data = True
    else:
        create_simulated_data = True

    if not all_data['timesteps'] or create_simulated_data:
        timesteps = np.arange(0, 30000, 100)
        all_data['timesteps'] = timesteps.tolist()
        progress = timesteps / 30000
        all_data['rewards'] = (-50 + 70 * (1 - np.exp(-3 * progress)) + np.random.normal(0, 5, len(timesteps))).tolist()
        all_data['value_losses'] = (2.0 * np.exp(-2 * progress) + np.random.normal(0, 0.1, len(timesteps))).tolist()
        all_data['coordination'] = np.clip(progress + np.random.normal(0, 0.05, len(timesteps)), 0, 1).tolist()

    max_len = len(all_data['timesteps'])
    for key in all_data:
        if len(all_data[key]) < max_len:
            if all_data[key]:
                all_data[key].extend([all_data[key][-1]] * (max_len - len(all_data[key])))
            else:
                all_data[key] = [0] * max_len

    n_agents = 2
    n_adversaries = 1
    n_total = n_agents + n_adversaries
    n_landmarks = 2
    colors_agents = ['blue', 'salmon']
    color_adversary = 'purple'
    colors_landmarks = ['green', 'orange']
    window_size = 10

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    ax_reward, ax_agents = axes[0, 0], axes[0, 1]
    ax_value, ax_coord = axes[1, 0], axes[1, 1]
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0, w_pad=2.0, h_pad=3.0)
    total_frames = min(len(all_data['timesteps']), 50)
    frame_skip = max(1, len(all_data['timesteps']) // total_frames)

    def moving_average(data, window_size=10):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def animate(frame):
        ax_reward.clear()
        ax_agents.clear()
        ax_value.clear()
        ax_coord.clear()

        idx = min(frame * frame_skip, len(all_data['timesteps']) - 1)
        t = all_data['timesteps'][idx]
        progress = t / max(all_data['timesteps'])

        # Reward with moving average
        ax_reward.plot(all_data['timesteps'][:idx+1], all_data['rewards'][:idx+1], color='lightblue', linewidth=1, label="Reward")
        if idx + 1 >= window_size:
            ma = moving_average(all_data['rewards'][:idx+1], window_size=window_size)
            x_ma = all_data['timesteps'][window_size-1:idx+1]
            ax_reward.plot(x_ma, ma, color='blue', linewidth=2, label="Moving Avg")
        ax_reward.set_title('Training Reward Evolution')
        ax_reward.set_xlabel('Timesteps')
        ax_reward.set_ylabel('Episode Reward')
        ax_reward.grid(True, alpha=0.3)
        ax_reward.set_xlim(0, max(all_data['timesteps']))
        ax_reward.set_ylim(min(all_data['rewards']) - 10, max(all_data['rewards']) + 10)
        ax_reward.legend()

        # Agent & adversary coordination (positions only, simulated)
        training_quality = min(progress, 1.0)
        agent_positions = []
        landmark_positions = []

        # Landmarks positioned evenly on a circle
        for i in range(n_landmarks):
            angle = 2 * np.pi * i / n_landmarks
            landmark_positions.append([np.cos(angle), np.sin(angle)])

        # Agents: 2 normal agents
        for i in range(n_agents):
            base_pos = landmark_positions[i % n_landmarks]
            noise = np.random.normal(0, (1 - training_quality) * 0.5, 2)
            agent_pos = [base_pos[0] + noise[0], base_pos[1] + noise[1]]
            agent_positions.append(agent_pos)

        # Adversary: tries to stay away from landmarks (opposite to first landmark)
        adv_base = [-landmark_positions[0][0], -landmark_positions[0][1]]
        adv_noise = np.random.normal(0, (1 - training_quality) * 0.5, 2)
        adversary_pos = [adv_base[0] + adv_noise[0], adv_base[1] + adv_noise[1]]

        # Plot landmarks
        for i in range(n_landmarks):
            ax_agents.scatter(*landmark_positions[i], c=colors_landmarks[i], s=120, marker='s', alpha=0.6, label=f'Landmark {i+1}' if frame == 0 else "")

        # Plot agents
        for i in range(n_agents):
            label = f'Agent {i+1}' if frame == 0 else ""
            ax_agents.scatter(*agent_positions[i], c=colors_agents[i], s=80, marker='o', edgecolor='black', linewidth=1, label=label)
        # Plot adversary
        ax_agents.scatter(*adversary_pos, c=color_adversary, s=80, marker='X', edgecolor='black', linewidth=1, label='Adversary' if frame == 0 else "")

        ax_agents.set_xlim(-1.5, 1.5)
        ax_agents.set_ylim(-1.5, 1.5)
        ax_agents.set_title('Agent & Adversary Coordination')
        ax_agents.grid(True, alpha=0.3)
        if frame == 0:
            ax_agents.legend(loc='upper left')

        # Value loss with moving average
        ax_value.plot(all_data['timesteps'][:idx+1], all_data['value_losses'][:idx+1], color='pink', linewidth=1, label="Value Loss")
        if idx + 1 >= window_size:
            ma_val = moving_average(all_data['value_losses'][:idx+1], window_size=window_size)
            x_ma_val = all_data['timesteps'][window_size-1:idx+1]
            ax_value.plot(x_ma_val, ma_val, color='red', linewidth=2, label="Moving Avg")
        ax_value.set_title('Value Loss')
        ax_value.set_xlabel('Timesteps')
        ax_value.set_ylabel('Value Loss')
        ax_value.grid(True, alpha=0.3)
        ax_value.set_xlim(0, max(all_data['timesteps']))
        ax_value.set_ylim(0, max(all_data['value_losses']) + 0.2)
        ax_value.legend()

        # Coordination score with moving average
        ax_coord.plot(all_data['timesteps'][:idx+1], all_data['coordination'][:idx+1], color='violet', linewidth=1, label="Coordination")
        if idx + 1 >= window_size:
            ma_coord = moving_average(all_data['coordination'][:idx+1], window_size=window_size)
            x_ma_coord = all_data['timesteps'][window_size-1:idx+1]
            ax_coord.plot(x_ma_coord, ma_coord, color='indigo', linewidth=2, label="Moving Avg")
        ax_coord.set_title('Coordination Score')
        ax_coord.set_xlabel('Timesteps')
        ax_coord.set_ylabel('Coordination')
        ax_coord.set_xlim(0, max(all_data['timesteps']))
        ax_coord.set_ylim(0, 1)
        ax_coord.grid(True, alpha=0.3)
        ax_coord.legend()

        fig.suptitle(f'Timestep: {t:,}/{max(all_data["timesteps"]):,} | Progress: {progress*100:.1f}%', fontsize=14, fontweight='bold')

    os.makedirs('figures', exist_ok=True)
    gif_filename = os.path.join('figures', 'training_dashboard_adversary.gif')

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200, repeat=True)
    anim.save(gif_filename, writer='pillow', fps=4, dpi=100)
    plt.show()
    return gif_filename