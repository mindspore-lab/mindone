ae_norm = {
    "CausalVAEModel_4x8x8": lambda x: 2.0 * x - 1.0,
}
ae_denorm = {
    "CausalVAEModel_4x8x8": lambda x: (x + 1.0) / 2.0,
}


def getdataset(args):
    raise NotImplementedError(args.dataset)
