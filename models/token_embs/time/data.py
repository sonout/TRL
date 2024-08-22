from torch.utils.data import Dataset
import numpy as np
import datetime
from tqdm import tqdm

class NextDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        return np.array(self.dates[idx]).astype(np.float32), np.array(self.dates[idx+1]).astype(np.float32)

class TimeDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in tqdm(dates)]

        self.dt_list = [datetime.datetime(year, month, day, hour, min, sec) for hour, min, sec, year, month, day in tqdm(self.dates)]
        self.dates = [[dt.hour, dt.minute, dt.second, dt.weekday()] for dt in tqdm(self.dt_list)]

        print("done")
        #self.dates = [[hour, min, sek, day] for hour, min, sek, year, month, day in self.dates]

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        x = np.array(self.dates[idx]).astype(np.float32)
        return x, x

if __name__ == "__main__":
    dt = open("/home/schestakov/projects/trajemb/models/proposed/time2vec/date_time.txt", 'r').readlines()
    dataset = TimeDateDataset(dt)
    print(dataset[0])