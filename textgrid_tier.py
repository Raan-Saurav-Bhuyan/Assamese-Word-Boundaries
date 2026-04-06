import textgrid

def print_tiers(textgrid_file):
    tg = textgrid.TextGrid.fromFile(textgrid_file)

    for i, tier in enumerate(tg):
        print(f"Tier {i}: {tier.name}")

    for interval in tg[1]:
        print(f"Interval min time = {interval.minTime}")
        print(f"Interval min time = {interval.maxTime}")
        print(f"Word = {interval.mark.strip()}")

if __name__ == "__main__":
    textgrid_file = "ASS/Textgrid/100_1_mono.TextGrid"
    print_tiers(textgrid_file)
