import webbrowser
from tkinter import *
from PIL import ImageTk, Image
from search_engine_best import SearchEngine
import os
import threading
PATH_FOR_DATA = bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')

class GUI:
    def __init__(self):
        self.todelete = []
        t1 = threading.Thread(target=self.load_se)
        t1.start()
        self.window_start()
        self.window_pref()

    def load_se(self):
        self.se = SearchEngine()
        self.se.build_index_from_parquet(PATH_FOR_DATA)
        self.window_pref()

    def window_start(self):
        self.window = Tk()
        self.window.title("Twitter Search Engine")
        self.window.geometry('800x700')
        img = ImageTk.PhotoImage(file="TSE.png")
        self.panel = Label(self.window, image=img)
        self.panel.pack(side="top", fill='both', expand="yes")
        self.lbl = Label(self.window, text='Loading Engine..')
        self.lbl.pack(side="bottom", expand="no")

        self.window.mainloop()

    def window_pref(self):
        try:
            self.lbl.destroy()
            self.panel.pack(side="top", expand="no")
            self.txt = Entry(self.window, width=60)
            self.txt.pack(side="top", expand="no", ipady=3)
            self.btn = Button(self.window, text="Start Searching", command=self.Search)
            self.btn.pack(side="top", expand="no", ipady=2, pady=5)

            self.window.update()
        except:
            pass

    def Search(self):
        query = self.txt.get()
        if query == '': return
        results = self.se.search(query)
        for item in self.todelete:
            item.destroy()
        self.lbl = Label(self.window, text='Results:')
        self.lbl.pack(side="top", expand="no")
        self.todelete.append(self.lbl)
        eval_link = lambda x: (lambda p: self.callback(f'https://twitter.com/anyuser/status/{x}'))
        for idx, tweet in enumerate(results[1][:20]):
            txt = f"Tweet rated #{idx+1} - Tweet ID: {tweet}"
            link1 = Label(self.window, text=txt, fg="blue", cursor="hand2")
            link1.pack()
            link1.bind("<Button-1>", eval_link(tweet))
            self.todelete.append(link1)

    def callback(self,url):
        webbrowser.open_new(url)


if __name__ == '__main__':
    g = GUI()