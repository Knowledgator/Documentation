# Installation

To begin using the GLiClass model, you can install the GLiClass Python library through pip, conda, or directly from the source.


## Install via Pip

```bash
pip install gliclass
```

## Install from Source

To install the GLiClass library from source, follow these steps:

1. **Clone the Repository:**

   First, clone the GLiClass repository from GitHub:

   ```bash
   git clone https://github.com/Knowledgator/GLiClass
   ```

2. **Navigate to the Project Directory:**

   Change to the directory containing the cloned repository:

   ```bash
   cd GLiClass
   ```

3. **Install Dependencies:**
   :::tip
   It's a good practice to create and activate a virtual environment before installing dependencies:
   :::

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

4. **Install the GLiNER Package:**

   Finally, install the GLiClass package using:

   ```bash
   pip install -U .
   ```
   :::tip
   Use ```pip install -U -e .``` to install in editable mode
   :::

5. **Verify Installation:**

   You can verify the installation by importing the library in a Python script:

   ```python
   import gliclass
   print(gliclass.__version__)
   ```
---