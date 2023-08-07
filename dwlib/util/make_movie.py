import imageio
import os

class make_movie():
    def __init__(self, path, extension):
        """
        path : directory where input images exist
        extension : extension of images. (jpeg, png, ...)
        """
        self._file_list = [f for f in os.listdir(path) if f.endswith(extension)]
    def to_gif(self, **kwargs):
        """
        path : directory to save file
        output : name of save file
        fps : fps of gif
        """
        path = kwargs["path"]+"/" if "path" in kwargs.keys() else self.path+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        with imageio.get_writer(path + kwargs["output"]+".gif", mode="I", duration=1/kwargs["fps"]) as writer:
            for filename in self._file_list:
                filename = os.path.join(path, filename)
                image = imageio.imread(filename)
                writer.append_data(image)
        return
