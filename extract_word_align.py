import textgrid
import os

def textgrid_to_align(textgrid_file, sample_rate, output_file):
    tg = textgrid.TextGrid.fromFile(textgrid_file)

    with open(output_file, 'w') as f:
        for interval in tg[1]:
            # Assuming 1 tier, adjust index if needed: --->
            start_time = interval.minTime
            end_time = interval.maxTime

            # Remove whitespace: --->
            word = interval.mark.strip()

            # skip empty intervals: --->
            if word:
                start_index = int(start_time * sample_rate)
                end_index = int(end_time * sample_rate)
                f.write(f"{start_index} {end_index} {word}\n")

if __name__ == "__main__":
    directory = "ASS/Textgrid"                  # current directory, change if needed
    sample_rate = 22050                           # adjust according to your audio files

    for filename in os.listdir(directory):
        if filename.endswith(".TextGrid"):
            textgrid_file = os.path.join(directory, filename)
            output_filename = filename.replace(".TextGrid", ".align")
            output_file = os.path.join(directory, output_filename)

            textgrid_to_align(textgrid_file, sample_rate, output_file)
            print(f"Generated {output_filename}")
