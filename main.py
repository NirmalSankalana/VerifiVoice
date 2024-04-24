from verifyvoice.DataLoader import DataLoader


sample1 = "./samples/dr-uthaya-e1.mp3"

s =  DataLoader.load_audio(sample1, 160)
print(s)