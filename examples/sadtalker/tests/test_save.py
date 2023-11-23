import imageio
import pickle

file = open("result.pkl", "rb")
data = pickle.load(file)
file.close()

save_path = "test.mp4"
imageio.mimsave(save_path, data,  fps=float(25))
