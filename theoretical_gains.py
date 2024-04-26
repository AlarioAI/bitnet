import matplotlib.pyplot as plt
import numpy as np


def plot_comparison_graph(image_sizes, combined_operations, additions):
    sizes_labels = [f"{w}x{h}" for w, h in image_sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_labels, combined_operations, label='Total Operations (Multiplication + Addition)', marker='o')
    plt.plot(sizes_labels, additions, label='Additions', marker='o')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Number of Operations')
    plt.title('Multiplication and Addition Operations in 3x3 Convolution')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)


    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    plt.gca().tick_params(colors='white', which='both')

    plt.show()


def MobileNetV2_operations():
    inverted_residual_settings = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    # Assume initial parameters for the input image and convolution
    input_height, input_width = 224, 224  # typical input dimensions for MobileNet
    input_channels = 3  # initial input channels (RGB image)

    # Function to calculate output dimensions after convolution
    def calculate_output_dim(input_dim, kernel_size, stride, padding=0):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    # Total operations calculation
    total_multiplications = 0
    total_additions = 0

    for t, c, n, s in inverted_residual_settings:
        for i in range(n):
            # Calculate the output dimensions for this block
            if i == 0:  # first layer in the block may have a stride
                out_height = calculate_output_dim(input_height, 3, s)
                out_width = calculate_output_dim(input_width, 3, s)
            else:  # remaining layers in the block will have a stride of 1
                out_height = calculate_output_dim(input_height, 3, 1)
                out_width = calculate_output_dim(input_width, 3, 1)

            # Depthwise Convolution
            depthwise_ops = out_height * out_width * 3 * 3 * input_channels
            pointwise_ops = out_height * out_width * input_channels * c

            # Update totals
            total_multiplications += depthwise_ops + pointwise_ops
            total_additions += depthwise_ops + pointwise_ops  # Each mult is typically followed by an add

            # Update input dimensions and channels for the next block
            input_height, input_width, input_channels = out_height, out_width, c


    print(f"Total Operations in MobileNetV2. Multiplications and Additions: {total_multiplications} and {total_additions} respectively.")


def main():
    initial_width = 32
    initial_height = 18
    max_width = 1920
    max_height = 1080

    image_sizes = [(initial_width * (2 ** i), initial_height * (2 ** i)) for i in range(int(np.log2(max_width / initial_width)) + 1)]

    kernel_size = 3

    operations = []
    for width, height in image_sizes:

        output_width = width - (kernel_size - 1)
        output_height = height - (kernel_size - 1)
        total_output_pixels = output_width * output_height

        multiplications_per_pixel = kernel_size ** 2

        additions_per_pixel = multiplications_per_pixel - 1

        total_multiplications = total_output_pixels * multiplications_per_pixel
        total_additions = total_output_pixels * additions_per_pixel
        operations.append((total_multiplications + total_additions, total_additions))


    combined_operations, additions = zip(*operations)
    ratio = additions[0] / combined_operations[0]
    print(f"Ratio of Additions to Multiplications: {ratio:.2f}")
    MobileNetV2_operations()
    plot_comparison_graph(image_sizes, combined_operations, additions)


if __name__ == '__main__':
    main()
