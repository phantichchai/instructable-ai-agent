import tkinter as tk
from inference import capture_from_window

def running():
    window_title = clicked.get()
    instruction = entry_instruction_text.get()
    capture_from_window.start_capturing(instruction, window_title, 16, 30)

# Create the main window
root = tk.Tk()
root.title("Inference Agent")
options = [
    "Genshin Impact",
    "Whthering Waves"
]
clicked = tk.StringVar()
clicked.set("Genshin Impact")

# Create and place the widgets
tk.Label(root, text="Window Title:").grid(row=0, column=0, padx=10, pady=10)
entry_window_title = tk.OptionMenu(root, clicked, *options)
entry_window_title.grid(row=0, column=1, columnspan=1, padx=10, pady=10)

tk.Label(root, text="Instruction:").grid(row=1, column=0, padx=10, pady=10)
entry_instruction_text = tk.Entry(root)
entry_instruction_text.grid(row=1, column=1, columnspan=1, padx=10, pady=10)

start_btn = tk.Button(root, text="Start", command=running)
start_btn.grid(row=1, column=2, columnspan=1, padx=10, pady=10)

root.mainloop()