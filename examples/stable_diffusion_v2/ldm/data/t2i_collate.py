from toolz.sandbox import unzip


data_column = [
    'img_feat',
    'txt_tokens'
]


def t2i_collate(inputs):
    """
    Return:
    :img_feat     (batch_size, height, weight, 3)
    :txt_tokens   (n, max_txt_len)
    """
    img_feat, txt_tokens = map(list, unzip(inputs))
    batch = {
        'img_feat': img_feat,
        'txt_tokens': txt_tokens,
    }
    return batch


data_column_db = [
    'train_img_feat',
    'train_txt_tokens',
    'reg_img_feat',
    'reg_txt_tokens'
]


def t2i_collate_db(inputs):
    """
    Return:
    :train_img_feat     (batch_size, height, weight, 3)
    :train_txt_tokens   (n, max_txt_len)
    :reg_img_feat     (batch_size, height, weight, 3)
    :reg_txt_tokens   (n, max_txt_len)
    """
    train_img_feat, train_txt_tokens, reg_img_feat, reg_txt_tokens= map(list, unzip(inputs))
    batch = {
        'train_img_feat': train_img_feat,
        'train_txt_tokens': train_txt_tokens,
        'reg_img_feat': reg_img_feat,
        'reg_txt_tokens': reg_txt_tokens,
    }
    return batch