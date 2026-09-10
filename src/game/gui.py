"""2048 图形界面（tkinter）。

支持两种模式：
- 人类游玩：方向键 上/下/左/右 控制。
- AI 演示：传入一个智能体(具有 select_action 方法)，点击"AI 自动"让其自动
  连续走子，可直观观察不同策略的下棋风格与效果。

tkinter 为 Python 标准库自带，无需额外安装。
"""
from __future__ import annotations

from typing import Optional

import tkinter as tk

from .board import Board, UP, DOWN, LEFT, RIGHT

# 不同数值方块的配色，贴近经典 2048 风格
TILE_COLORS = {
    0: ("#cdc1b4", ""),
    2: ("#eee4da", "#776e65"),
    4: ("#ede0c8", "#776e65"),
    8: ("#f2b179", "#f9f6f2"),
    16: ("#f59563", "#f9f6f2"),
    32: ("#f67c5f", "#f9f6f2"),
    64: ("#f65e3b", "#f9f6f2"),
    128: ("#edcf72", "#f9f6f2"),
    256: ("#edcc61", "#f9f6f2"),
    512: ("#edc850", "#f9f6f2"),
    1024: ("#edc53f", "#f9f6f2"),
    2048: ("#edc22e", "#f9f6f2"),
}
DEFAULT_TILE = ("#3c3a32", "#f9f6f2")


class Game2048GUI:
    def __init__(self, agent=None, ai_delay_ms: int = 120):
        self.agent = agent
        self.ai_delay_ms = ai_delay_ms
        self.ai_running = False

        self.root = tk.Tk()
        self.root.title("2048 - 人工智能大作业")
        self.root.configure(bg="#bbada0")
        self.board = Board()

        top = tk.Frame(self.root, bg="#bbada0")
        top.pack(padx=10, pady=10, fill="x")
        self.score_var = tk.StringVar()
        tk.Label(top, textvariable=self.score_var, font=("Arial", 16, "bold"),
                 bg="#bbada0", fg="white").pack(side="left")
        tk.Button(top, text="新游戏", command=self.new_game).pack(side="right", padx=4)
        if agent is not None:
            self.ai_btn = tk.Button(top, text="AI 自动", command=self.toggle_ai)
            self.ai_btn.pack(side="right", padx=4)
            tk.Button(top, text="AI 单步", command=self.ai_step).pack(side="right", padx=4)

        grid = tk.Frame(self.root, bg="#bbada0")
        grid.pack(padx=10, pady=(0, 10))
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                lbl = tk.Label(grid, text="", width=4, height=2,
                               font=("Arial", 24, "bold"), bg=TILE_COLORS[0][0])
                lbl.grid(row=i, column=j, padx=5, pady=5, ipadx=6, ipady=6)
                row.append(lbl)
            self.cells.append(row)

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 12),
                 bg="#bbada0", fg="#f9f6f2").pack(pady=(0, 10))

        self.root.bind("<Up>", lambda e: self.human_move(UP))
        self.root.bind("<Down>", lambda e: self.human_move(DOWN))
        self.root.bind("<Left>", lambda e: self.human_move(LEFT))
        self.root.bind("<Right>", lambda e: self.human_move(RIGHT))
        self.refresh()

    def new_game(self):
        self.ai_running = False
        self.board = Board()
        self.status_var.set("")
        self.refresh()

    def human_move(self, action: int):
        if self.ai_running:
            return
        self._do_move(action)

    def _do_move(self, action: int):
        moved, _ = self.board.move(action)
        self.refresh()
        if self.board.is_game_over():
            self.status_var.set("游戏结束！")
            self.ai_running = False

    def ai_step(self):
        if self.agent is None or self.board.is_game_over():
            return
        a = self.agent.select_action(self.board)
        if a is not None:
            self._do_move(a)

    def toggle_ai(self):
        if self.agent is None:
            return
        self.ai_running = not self.ai_running
        self.ai_btn.config(text="AI 暂停" if self.ai_running else "AI 自动")
        if self.ai_running:
            self._ai_loop()

    def _ai_loop(self):
        if not self.ai_running or self.board.is_game_over():
            self.ai_running = False
            if self.board.is_game_over():
                self.status_var.set("游戏结束！")
            return
        a = self.agent.select_action(self.board)
        if a is None:
            self.ai_running = False
            return
        self._do_move(a)
        self.root.after(self.ai_delay_ms, self._ai_loop)

    def refresh(self):
        self.score_var.set(f"得分: {self.board.score}    最大: {self.board.max_tile()}")
        for i in range(4):
            for j in range(4):
                v = int(self.board.grid[i, j])
                bg, fg = TILE_COLORS.get(v, DEFAULT_TILE)
    #            self.cells[i][j].config(text="" if v == 0 else str(v), bg=bg, fg=fg)
                # 修改后的代码
                if v == 0:
                    self.cells[i][j].config(text="", bg="#cdc1b4", fg="#776e65")
                else:
                    self.cells[i][j].config(text=str(v), bg=bg, fg=fg)

    def run(self):
        self.root.mainloop()
