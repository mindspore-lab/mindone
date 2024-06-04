full_width2b = ["……", "。。。", "。", "，", "；", "：", "？", "！", "“", "”", "‘", "’", "（", "）", "【", "】", "、"]
half_width = ["...", "...", ".", ",", ";", ":", "?", "!", '"', '"', "'", "'", "(", ")", "[", "]", ","]

full_width = ["……", "。。。", "。", "，", "；", "：", "？", "！", "（", "）", "【", "】"]
half_width2q = ["...", "...", ".", ",", ";", ":", "?", "!", "(", ")", "[", "]"]


def repl(data, from_width, to_width):
    assert len(from_width) == len(to_width)
    for i, j in zip(from_width, to_width):
        data = data.replace(i, j)
    return data


def process(line):
    new_line = line.replace("\n", " ")
    p1 = [",", ".", ";", ":", "?", "!"]
    for _ in range(5):
        for p in p1:
            new_line = new_line.replace(" " + p + " ", p)
            new_line = new_line.replace(p + " ", p)
            new_line = new_line.replace(" " + p, p)

    for p in p1:
        new_line = new_line.replace(p, p + " ")

    new_line = new_line.replace(". . . ", "... ")

    wrong_samples = []
    for i in range(1, len(new_line) - 2):
        if new_line[i] == "'" and new_line[i + 1].isalpha() and new_line[i - 1] == " " and new_line[i + 2] == " ":
            j = i - 2
            while j >= 1 and new_line[j] == " ":
                j -= 1
            wrong_samples.append(new_line[j : i + 3])

    wrong_samples.sort(key=lambda x: len(x), reverse=True)
    for w in wrong_samples:
        new_line = new_line.replace(w, w[0] + w[-3:])
    new_line = new_line.replace(" n't", "n't")
    for k in range(len(new_line) - 1, -1, -1):
        if new_line[k] != " ":
            new_line = new_line[: k + 1]
            break
    return new_line


def en_cleaning(data):
    d = repl(data, full_width2b, half_width)
    d = process(d)
    new_d = d.replace("  ", " ")
    while new_d != d:
        d = new_d
        new_d = d.replace("  ", " ")
    return new_d


def zh_cleaning(data):
    d = repl(data, half_width2q, full_width)
    d = d.replace(" ", "")
    d = d.replace("\n", "")
    return d
