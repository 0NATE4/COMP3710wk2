# Demo Day Plan: COMP3710 Lab 1

A concise checklist for the lab demonstration, covering all marking criteria.

---

## Part 1: Gaussian & Gabor (2 Marks)

### What to Show
1. Run the script to show the **2D sine wave**.
2. Run the script to show the **Gabor filter** (the sine wave multiplied by the Gaussian).

### What to Say
- **(1 Mark):** "Here is the 2D sine function I created, as required."
- **(1 Mark):** "And here is the result of multiplying the sine and Gaussian tensors together, which creates the Gabor filter."

---

## Part 2: Mandelbrot & Julia (2 Marks)

### What to Show
1. Run `python your_script.py --mandelbrot`. Zoom into a detailed area to demonstrate high resolution.
2. Run `python your_script.py --julia`.

### What to Say
- **(1 Mark - High-Res):** "Here is the Mandelbrot set. My code fulfills the high-resolution requirement by **dynamically re-computing** the fractal at the window's full pixel resolution every time I zoom, preventing pixelation."
- **(1 Mark - Julia Set):** "And by changing the command-line flag, the same code now generates the Julia set."

---

## Part 3: Your Fractal & Justification (4 Marks)

### What to Show
1. Run `python your_script.py --newton` to show your custom fractal (the Newton fractal).
2. Have your GitHub repository open in a browser, logged in, showing the code.

### What to Say

- **(1 Mark - GitHub):**
  > "Here is the final code available in my GitHub repository."

- **(3 Marks - Justification):**
  1. **Major Component:**
     > "My program uses PyTorch as its **core engine**. The entire grid of pixels is loaded into a single PyTorch tensor."
  2. **Parallelism:**
     > "This allows for massive **parallelism**. One line of code, like the Newton's method formula, performs hundreds of thousands of calculations simultaneously on the GPU."
  3. **Reasonable Way:**
     > "This is a **reasonable and effective** use of parallelism because it makes real-time, interactive exploration possible, which would be impossible with traditional loops."

---
. Why We Chose This
"As my substantial analysis, I chose to visualize the basins of attraction for the Newton fractal. I picked this because it's more than just a cosmetic change; it analyzes the fundamental mathematical behavior of the system. It directly answers the question: 'Where do all the points end up?'"

2. What It Means
"The Newton fractal is based on finding the three solutions to the equation z³ - 1 = 0. Every starting point on the screen eventually falls toward one of these three solutions. A 'basin of attraction' is the entire region of points that are all pulled toward the same solution.
As you can see, the three main colored regions—red, green, and blue—are the basins for each of the three roots."

3. How We Did It
"My code implements this in two steps:
It first runs the standard Newton iteration for every pixel.
Then, it checks which of the three roots the final point is closest to and assigns a primary color—red, green, or blue—based on that result.
The dark, turbulent boundaries are where points take a long time to 'decide' which basin to fall into, and that's where the fractal structure lies."