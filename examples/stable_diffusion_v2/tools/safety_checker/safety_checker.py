class SafetyChecker:
    """
    Abstract class for different versions of safety checker
    """
    def eval_safety(self, special_sim, nsfw_sim):
        raise NotImplementedError
    
    def eval(self, paths):
        raise NotImplementedError
    
    def __call__(self, images):
        raise NotImplementedError


if __name__ == "__main__":
    print(dir(SafetyChecker()))
