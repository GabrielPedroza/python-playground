def main(a, b):
    output = []

    p1 = len(a) - 1
    p2 = len(b) - 1

    while p1 >= 0 and p2 >= 0:
        total = int(a[p1]) + int(b[p2])
        output.append(str(total))
        p1 -= 1
        p2 -= 1

    if p1 >= 0:
        output.insert(0, a[:p1 + 1])

    if p2 >= 0:
        output.insert(0, b[:p2 + 1])

    print("".join(output))


if __name__ == "__main__":
    # main("99", "99") # 1818
    main("11", "9") # 110
    # p1 = 1
    # p2 = 0
    # output = ["110"]
