import tkinter as tk
from tkinter import ttk
import numpy as np
from ctypes import windll

# Fonts
TIMES12BOLD = ("Times New Roman", 12, "bold")
TIMES14BOLD = ("Times New Roman", 14, "bold")
TIMES16BOLD = ("Times New Roman", 16, "bold")

DEFAULT_BTN = ("Times New Roman", 11)

class MDPConstructionGUI:
    """
    This is a GUI form that collects the neccessary information for an MDP and stores the user inputs in the public field inputs.
    Calling its constructor invokes the form automatically.

    Paramaters
    ----------
    inputs: Dict with length 6
        ["S"] = Number of states (Int)
        ["A"] = Number of actions (Int)
        ["R"] = Number of rewards (Int)
        ["Rewards"] = Possible reward values (List[Float])
        ["RewardTriples"] = NumPy array of floats of shape (S, A, S, R), representing the cur_state-action-next_state reward triples probability distribution
        ["Transitions"] = NumPy array of floats of shape (S, A, S), representing the cur_state-action state transitions probability distribution
    """
    
    def __init__(self):
        windll.shcore.SetProcessDpiAwareness(1)
        self.inputs = {"S" : None, "A" : None, "R" : None, "Transitions" : None, "Rewards" : None, "RewardTriples" : None}
        self._window = tk.Tk()
        self._draw_initial(self._window)
        self._main_frame = ttk.Frame()
        self._main_frame.pack()
        self._draw_S_selection()
        self._window.mainloop()
    
    def _draw_initial(self, window):
        width = self._window.winfo_screenwidth()
        height = self._window.winfo_screenheight()
        self._window.geometry(str(width) + "x" + str(height))
        self._window.title("Input for MDP")
    
    def _draw_S_selection(self):
        S_selection_frame = tk.Frame(master=self._main_frame, pady=10)
        S_selection_frame.pack()
        S_lbl = tk.Label(S_selection_frame, text="How many states?", font=TIMES12BOLD)
        S_lbl.grid(row=0, column=0, padx=5)
        S_entry = tk.Entry(S_selection_frame)
        S_entry.grid(row=0, column=1, padx=5)
        S_button = tk.Button(S_selection_frame, text="Confirm", 
        command= lambda: self._get_S(S_entry, S_selection_frame), font=DEFAULT_BTN)
        S_button.grid(row=0, column=2, padx=5)
    
    def _get_S(self, S_entry, S_frame):
        S = S_entry.get()
        try:
            int(S)
            print("You inputted " + S)
        except ValueError:
            print("You provided " + S + " which isn't an integer.")
            return
        
        self.inputs["S"] = int(S)
        self._draw_A_selection()
        S_frame.pack_forget()
    
    def _draw_A_selection(self):
        A_selection_frame = tk.Frame(master=self._main_frame, pady=10)
        A_selection_frame.pack()
        A_lbl = tk.Label(A_selection_frame, text="How many actions?", font=TIMES12BOLD)
        A_lbl.grid(row=0, column=0, padx=5)
        A_entry = tk.Entry(A_selection_frame)
        A_entry.grid(row=0, column=1, padx=5)
        A_button = tk.Button(A_selection_frame, text="Confirm", 
        command= lambda: self._get_A(A_entry, A_selection_frame), font=DEFAULT_BTN)
        A_button.grid(row=0, column=2, padx=5)
    
    def _get_A(self, A_entry, A_frame):
        A = A_entry.get()
        try:
            int(A)
            print("You inputted " + A)
        except ValueError:
            print("You provided " + A + " which isn't an integer.")
            return

        self.inputs["A"] = int(A)
        self._draw_state_transitions_selection()
        A_frame.pack_forget()
    
    def _draw_state_transitions_selection(self):
        transitions_block_frame = tk.Frame(master=self._main_frame, pady=10)
        transitions_block_frame.pack()
        
        transitions_lbl = tk.Label(transitions_block_frame, 
        text="The P(s' | a, s) function:", font=TIMES16BOLD, pady=10)
        transitions_lbl.pack()

        state_transitions_frame = tk.Frame(master=transitions_block_frame)
        state_transitions_frame.pack(pady=10)
        S = self.inputs["S"]
        A = self.inputs["A"]
        transition_entries = np.zeros(shape=(S, A, S), dtype=object)

        for s in range(S):
            state_frame = tk.Frame(master=state_transitions_frame, pady=10, bd=3, relief=tk.GROOVE)
            state_frame.pack()
            
            current_state_label = tk.Label(master=state_frame, text="s = " + str(s), font=TIMES14BOLD)
            current_state_label.pack()
            
            next_state_label = tk.Label(master=state_frame, text="s'", font=TIMES12BOLD)
            next_state_label.pack()
            
            matrix_area_frame = tk.Frame(master=state_frame)
            matrix_area_frame.pack(side=tk.LEFT)
            
            action_label = tk.Label(master=matrix_area_frame, text="a", font=TIMES12BOLD)
            action_label.grid(row=0, column=0, padx=5)

            actual_matrix = tk.Frame(master=matrix_area_frame)
            actual_matrix.grid(row=0, column=1, padx=(0, 10))

            for i in range(A+1):
                for j in range(S+1):
                    if (i == 0 and j == 0):
                        continue
                    elif (i == 0 or j == 0):
                        label = tk.Label(master=actual_matrix, text=str(max(i, j)-1))
                        label.grid(row=i, column=j)
                    else:
                        elem_frame = tk.Frame(master=actual_matrix, relief=tk.RAISED, borderwidth=1)
                        elem_frame.grid(row=i, column=j)
                        default_value = tk.StringVar(elem_frame, value='0')
                        entry = tk.Entry(elem_frame, textvariable=default_value, width=4)
                        transition_entries[s,i-1,j-1] = entry
                        entry.pack()
        
        submit_btn = tk.Button(master=state_transitions_frame, 
                               text="Confirm", 
                               command= lambda: self._get_transitions(transition_entries, transitions_block_frame),
                               font=DEFAULT_BTN)
        submit_btn.pack(pady=10)
    
    def _get_transitions(self, transition_entries, transitions_block_frame):
        get = np.vectorize(lambda entry: entry.get())
        transitions = get(transition_entries).astype(np.float)
        self.inputs["Transitions"] = transitions
        self._draw_rewards_selection()
        transitions_block_frame.pack_forget()
    
    def _draw_rewards_selection(self):
        rewards_frame = tk.Frame(master=self._main_frame, pady=10)
        rewards_frame.pack()
        rewards_lbl = tk.Label(master=rewards_frame, text="Possible reward values, comma-separated:", font=TIMES12BOLD)
        rewards_lbl.grid(row=0, column=0, padx=5)
        rewards_entry = tk.Entry(master=rewards_frame)
        rewards_entry.grid(row=0, column=1, padx=5)
        rewards_button = tk.Button(master=rewards_frame, text="Confirm", 
                                   command= lambda: self._get_rewards(rewards_entry, rewards_frame),
                                   font=DEFAULT_BTN)
        rewards_button.grid(row=0, column=2, padx=5)
    
    def _get_rewards(self, rewards_entry, rewards_frame):
        rewards = [float(reward) for reward in rewards_entry.get().split(',')]
        rewards = sorted(rewards)
        self.inputs["Rewards"] = rewards
        self.inputs["R"] = len(rewards)
        self._draw_reward_sas_triples_selection()
        rewards_frame.pack_forget()
    
    def _draw_reward_sas_triples_selection(self):
        triples_block_frame = tk.Frame(master=self._main_frame, pady=10)
        triples_block_frame.pack()
        triples_lbl = tk.Label(triples_block_frame, 
                               text="The P(r | s', a, s) function:",
                               font=TIMES16BOLD,
                               pady=20)
        triples_lbl.pack()
        triples_frame = tk.Frame(master=triples_block_frame)
        triples_frame.pack()
        S = self.inputs["S"]
        A = self.inputs["A"]
        R = self.inputs["R"]
        rewards = self.inputs["Rewards"]
        reward_entries = np.zeros(shape=(S, A, S, R), dtype=object)

        for s in range(S):
            state_frame = tk.Frame(master=triples_frame, pady=10, bd=3, relief=tk.GROOVE)
            state_frame.pack()
            
            current_state_label = tk.Label(master=state_frame, text="s = " + str(s) + ":", font=TIMES14BOLD)
            current_state_label.grid(row=s, column=0, padx=5)
            
            matrix_list_frame = tk.Frame(master=state_frame)
            matrix_list_frame.grid(row=s, column=1)

            for a in range(A):
                matrix_area_frame = tk.Frame(master=matrix_list_frame)
                matrix_area_frame.grid(row=0, column=a, padx=10)
                
                reward_label = tk.Label(master=matrix_area_frame, text="r", font=TIMES12BOLD)
                reward_label.grid(row=0, column=1)

                state_label = tk.Label(master=matrix_area_frame, text="s'", font=TIMES12BOLD)
                state_label.grid(row=1, column=0, padx=5)

                action_label = tk.Label(master=matrix_area_frame, text="a = " + str(a), font=TIMES14BOLD)
                action_label.grid(row=2, column=1)

                actual_matrix = tk.Frame(master=matrix_area_frame)
                actual_matrix.grid(row=1, column=1)

                for i in range(S+1):
                    for j in range(R+1):
                        if (i == 0 and j == 0):
                            continue
                        elif i == 0:
                            label = tk.Label(master=actual_matrix, text= str(rewards[j - 1]))
                            label.grid(row=i, column=j)
                        elif j == 0:
                            label = tk.Label(master=actual_matrix, text= str(i - 1))
                            label.grid(row=i, column=j)
                        else:
                            elem_frame = tk.Frame(master=actual_matrix, relief=tk.RAISED, borderwidth=1)
                            elem_frame.grid(row=i, column=j)
                            default_value = tk.StringVar(elem_frame, value='0')
                            entry = tk.Entry(elem_frame, textvariable=default_value, width=4)
                            reward_entries[s,a, i-1,j-1] = entry
                            entry.pack()
        
        submit_btn = tk.Button(master=triples_frame, 
                               text="Confirm", 
                               command= lambda: self._get_reward_sas_triples(reward_entries, triples_block_frame))
        submit_btn.pack(pady=10)
    
    def _get_reward_sas_triples(self, reward_entries, triples_block_frame):
        get = np.vectorize(lambda entry: entry.get())
        reward_triples = get(reward_entries).astype(np.float)
        self.inputs["RewardTriples"] = reward_triples
        print(reward_triples)
        triples_block_frame.pack_forget()
        self._close()
    
    def _close(self):
        self._window.destroy()

MDPConstructionGUI()