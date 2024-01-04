from PIL import Image
import numpy as np
from collections import defaultdict
from flask import Flask, render_template, request
import random


def load_image_to_array(image_path):
    # Open the image file
    img = Image.open(image_path)
    # Ensure the image is in RGB format
    img = img.convert('RGB')
    # Convert the image to an array
    img_array = np.array(img)

    return img_array
class Creature:
    # Class variables for weights
    color_weight = {
        'A': 1.5,  # Dark Brown
        'B': 0.8,  # Light Brown
        'C': 0.7,  # Light Purple
        'D': 1.0,  # Blue
        'E': 0.6,  # Brown
        'F': 0.65, # Light Brown with Light Purple
        'G': 0.5,  # Light Gray
        'J': 1.3,  # Dark Blue
        'K': 1.4,  # Dark Purple
        'L': 1.2   # Dark Brown
    }
    size_weight = {1: 0.8, 2: 1.0, 3: 1.0, 4: 0.8}
    limbs_weight = {1: 0.5, 2: 0.8, 3: 1.0, 4: 1.2, 5: 1.2, 6: 1.0, 7: 0.8, 8: 0.6, 9: 0.4, 10: 0.2}
    height_weight = {1: 0.8, 2: 1.2, 3: 1.0}
    # Color map as a class variable
    color_map = {
        'A': 'PURPLE',
        'B': 'LIGHT BROWN',
        'C': 'LIGHT PURPLE',
        'D': 'BLUE',
        'E': 'BROWN',
        'F': 'LIGHT BROWN WITH LIGHT PURPLE',
        'G': 'LIGHT GRAY',
        'J': 'DARK BLUE',
        'K': 'DARK PURPLE',
        'L': 'DARK BROWN'
    }
    color_luminance = {
        'A': 0.1,  # Dark Brown
        'B': 0.3,  # Light Brown
        'C': 0.7,  # Light Purple
        'D': 0.6,  # Blue
        'E': 0.4,  # Brown
        'F': 0.5,  # Light Brown with Light Purple
        'G': 0.9,  # Light Gray
        'J': 0.2,  # Dark Blue
        'K': 0.1,  # Dark Purple
        'L': 0.3   # Dark Brown
    }

    def __init__(self, image_array, color_code, size, limbs, height):
        self.image_array = image_array
        self.color_code = color_code
        self.color_description = Creature.color_map[color_code]  # Get color description from color map
        self.size = size
        self.limbs = limbs
        self.height = height
        self.genome = (color_code, size, limbs, height)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self, environment_factors=None, time_of_day='daytime'):
        """
        Calculate the creature's fitness score, optionally taking into account different environmental factors.
        :param environment_factors: Dictionary with keys as environmental factors (e.g., 'predator', 'food', 'mate') and values indicating whether they are active.
        """
        # Base fitness calculation using class variables for weights
        base_fitness = (Creature.color_weight[self.color_code] +
                        Creature.size_weight[self.size] +
                        Creature.limbs_weight[self.limbs] +
                        Creature.height_weight[self.height])
        
        # Simple noise estimator: high-frequency pixel changes
        noise_level = np.sum(np.abs(np.diff(self.image_array, axis=0))) + np.sum(np.abs(np.diff(self.image_array, axis=1)))
        noise_penalty_factor = 0.001  # This is an arbitrary value; adjust based on your needs
        noise_penalty = noise_penalty_factor * noise_level
        
        # Environmental factor adjustments
        environmental_fitness = 0
        if environment_factors:
            if environment_factors.get('predator'):
                # Predator avoidance might depend on color camouflage and agility (limbs)
                environmental_fitness += self.calculate_predator_avoidance(time_of_day)
            if environment_factors.get('food'):
                # Food gathering might depend on size and limbs
                environmental_fitness += self.calculate_food_gathering()
            if environment_factors.get('mate'):
                # Mating success might depend on color and height
                environmental_fitness += self.calculate_mating_success()

        # Combine base fitness with environmental adjustments and subtract noise penalty
        total_fitness = base_fitness + environmental_fitness - noise_penalty
        return total_fitness

    def calculate_predator_avoidance(self, time_of_day):
        """
        Calculate the predator avoidance score based on time of day.
        """
        # Get the luminance value for the creature's color
        creature_luminance = Creature.color_luminance[self.color_code]
        
        # Determine if the creature's color is advantageous based on time of day
        if time_of_day == 'daytime':
            # Lighter colors have an advantage during the day
            luminance_score = creature_luminance
        else:
            # Darker colors have an advantage during the night
            luminance_score = 1 - creature_luminance
        
        # Combine the luminance score with the limbs score for agility
        avoidance_score = luminance_score * Creature.color_weight[self.color_code] + Creature.limbs_weight[self.limbs]
        return avoidance_score
    
    def calculate_food_gathering(self):
        # Assume some logic to calculate food gathering based on size and limbs
        return Creature.size_weight[self.size] + Creature.limbs_weight[self.limbs]

    def calculate_mating_success(self):
        # Assume some logic to calculate mating success based on color and height
        return Creature.color_weight[self.color_code] + Creature.height_weight[self.height]

# Function to load creature images and create creature instances
def create_creatures():
    # Define the base path for the image files
    base_image_path = "C:/Users/PC1/OneDrive/Documents/MonsterMash/static/web/static/images"
    creatures_list = []


    # Assigning the color sequence to a list
    color_sequence = "ABBBGDAEDEFBADGGGAAEEDGGGGGGEEEEJJJJJKKJJJJLLLLLLLKLKJKJKJJJJ"

    # Size, limbs, and height sequences as lists
    size_sequence = "3333333334444422222222222222222211111111111221211333322222222"
    limbs_sequence = "8589668668666566668666666444444434333544444546466556565858888"
    height_sequence = "2332333322222212212112222222222221331333223231322333313332222"

    # Check if all sequences are of the same length to avoid errors
    assert len(color_sequence) == len(size_sequence) == len(limbs_sequence) == len(height_sequence) == 61

    # Generate the list of creatures


    for i in range(61):
        creature_number = i + 1
        image_path = f"{base_image_path}/Creature{creature_number}.jpg"
        image_array = load_image_to_array(image_path)  # Load the image as an array
        color_code = color_sequence[i]
        size = int(size_sequence[i])
        limbs = int(limbs_sequence[i])
        height = int(height_sequence[i])

        # Create a new instance of Creature
        creature = Creature(image_array, color_code, size, limbs, height)
        creatures_list.append(creature)    
    return creatures_list
creatures_list = create_creatures()

def tournament_selection(creatures, tournament_size=3):
    # Randomly select creatures for the tournament
    selected_for_tournament = random.sample(creatures, tournament_size)
    
    # Select the best creature from the tournament
    winner = max(selected_for_tournament, key=lambda x: x.fitness)
    return winner

# Example of selecting two parents for crossover
parent1 = tournament_selection(creatures_list)
parent2 = tournament_selection(creatures_list)

# Ensure parent1 and parent2 are not the same instance
while parent1 == parent2:
    parent2 = tournament_selection(creatures_list)

def save_array_as_image(img_array, save_path):
    img = Image.fromarray(img_array)
    img.save(save_path)

def crossover(parent1, parent2, blend_height=10):
    assert parent1.image_array.shape == parent2.image_array.shape
    height, width, channels = parent1.image_array.shape

    new_image_array = np.zeros((height, width, channels), dtype=np.uint8)

    # Define the region where the crossover occurs (middle of the image)
    blend_start = (height // 2) - (blend_height // 2)
    blend_end = (height // 2) + (blend_height // 2)

    # Top half from parent 1
    new_image_array[:blend_start, :] = parent1.image_array[:blend_start, :]

    # Blend region from both parents
    for i in range(blend_height):
        alpha = i / blend_height
        new_image_array[blend_start + i, :] = (1 - alpha) * parent1.image_array[blend_start + i, :] + alpha * parent2.image_array[blend_start + i, :]

    # Bottom half from parent 2
    new_image_array[blend_end:, :] = parent2.image_array[blend_end:, :]

    offspring = Creature(new_image_array, *random.choice([parent1.genome, parent2.genome]))
    return offspring

def mutate(creature, mutation_rate=0.01):
    img = creature.image_array.copy()
    height, width, channels = img.shape

    # Apply mutation across the whole image
    for i in range(height):
        for j in range(width):
            if np.random.rand() < mutation_rate:
                # Randomly change the color of the pixel
                img[i, j] = np.random.randint(0, 256, 3)

    mutated_creature = Creature(img, *creature.genome)
    return mutated_creature

def select_parents(creatures, desired_color, desired_size, desired_limbs, desired_height):
    # Filter creatures by the exact attributes
    # print(desired_color,desired_height,desired_limbs,desired_size)
    candidates = [creature for creature in creatures if creature.color_code == desired_color
                  and creature.size == desired_size
                  and creature.limbs == desired_limbs
                  and creature.height == desired_height]

    # If there are not enough candidates for a diverse selection, widen the criteria
    if len(candidates) < 1:
        print("not matched")

        # If no creatures match exactly, select creatures with the closest size and limbs, for example
        candidates = sorted(creatures, key=lambda x: (abs(x.size - desired_size) + abs(x.limbs - desired_limbs)))

    # Select two parents, ensuring they are not the same creature
    # parent1 = candidates[0]
    parent = candidates[0] if len(candidates) >= 1 else random.choice(creatures)

    print("parent",parent.color_code,parent.size,parent.limbs,parent.height)
    # print("parent2",parent2.color_code,parent2.size,parent2.limbs,parent2.height)

    return parent

# Initialize environmental settings
environmental_factors = {'predator': False, 'food': False, 'mate': False}
time_of_day = 'daytime'  # Default value

creatures = []  # This would be a list of Creature instances

# Create the Flask application instance
app = Flask(__name__)
# Sequences from your code
color_sequence = "ABBBGDAEDEFBADGGGAAEEDGGGGGGEEEEJJJJJKKJJJJLLLLLLLKLKJKJKJJJJ"
size_sequence = "3333333334444422222222222222222211111111111221211333322222222"
limbs_sequence = "8589668668666566668666666444444434333544444546466556565858888"
height_sequence = "2332333322222212212112222222222221331333223231322333313332222"

# Create default dictionaries to hold the attributes for each color
color_to_sizes = defaultdict(set)
color_to_limbs = defaultdict(set)
color_to_heights = defaultdict(set)

# Iterate over the sequences and populate the dictionaries
for i, color in enumerate(color_sequence):
    color_to_sizes[color].add(size_sequence[i])
    color_to_limbs[color].add(limbs_sequence[i])
    color_to_heights[color].add(height_sequence[i])

# Convert sets to lists for JSON serialization
color_to_sizes = {k: list(v) for k, v in color_to_sizes.items()}
color_to_limbs = {k: list(v) for k, v in color_to_limbs.items()}
color_to_heights = {k: list(v) for k, v in color_to_heights.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    global creatures_list, generated_creatures
    if request.method == 'GET':
        # Extract unique attributes for each category
        unique_colors = sorted({creature.color_code for creature in creatures_list})
        unique_sizes = sorted({creature.size for creature in creatures_list})
        unique_limbs = sorted({creature.limbs for creature in creatures_list})
        unique_heights = sorted({creature.height for creature in creatures_list})
        return render_template('index.html', color_to_sizes=color_to_sizes, color_to_limbs=color_to_limbs, color_to_heights=color_to_heights,
                           unique_sizes=unique_sizes,unique_limbs=unique_limbs,unique_heights=unique_heights)   
    # global creatures_list, generated_creatures
    selected_parents = [None, None]
    generated_creature = None
    # Extract unique attributes for each category
    unique_colors = sorted({creature.color_code for creature in creatures_list})
    unique_sizes = sorted({creature.size for creature in creatures_list})
    unique_limbs = sorted({creature.limbs for creature in creatures_list})
    unique_heights = sorted({creature.height for creature in creatures_list})

 
    if request.method == 'POST':

        # Extract common color from the form
        common_color_code = request.form['parent1_color']  # Since both colors are synced

        # Extract other attributes for each parent from the form
        parent1_attrs = {
            'color_code': common_color_code,
            'size': int(request.form['parent1_size']),
            'limbs': int(request.form['parent1_limbs']),
            'height': int(request.form['parent1_height'])
        }
        parent2_attrs = {
            'color_code': common_color_code,
            'size': int(request.form['parent2_size']),
            'limbs': int(request.form['parent2_limbs']),
            'height': int(request.form['parent2_height'])
        }

        # Environmental factors
        environmental_factors = {
            'predator': request.form.get('predator') == 'on',
            'food': request.form.get('food') == 'on',
            'mate': request.form.get('mate') == 'on'
        }
        time_of_day = request.form.get('time_of_day')
        result1 = find_creature_by_attributes(parent1_attrs)
        result2 = find_creature_by_attributes(parent2_attrs)
        parent1=result1[0] if result1 else None
        parent2=result2[0] if result2 else None
        selected_parents = [parent1, parent2] if parent1 and parent2 else [None, None]
        # Find and store the selected parent creatures

        # Run the GA loop with the selected parents and factors
        if parent1 and parent2:
            
            import os
            absolute_path = result1[1] 
            static_folder = "C:/Users/PC1/OneDrive/Documents/MonsterMash/static"
            images_paths=[]
    # Get the relative path
            parent1_image_path = os.path.relpath(absolute_path, static_folder) 
            images_paths.append(parent1_image_path)

            absolute_path = result2[1]  
            static_folder = "C:/Users/PC1/OneDrive/Documents/MonsterMash/static"

    # Get the relative path
            parent2_image_path = os.path.relpath(absolute_path, static_folder) 
            images_paths.append(parent2_image_path)

            generated_creature = generate_offspring(parent1_attrs, parent2_attrs, environmental_factors, time_of_day)
            print("got",generated_creature.image_path)
            

            absolute_path = generated_creature.image_path   
            static_folder = "C:/Users/PC1/OneDrive/Documents/MonsterMash/static"

# Get the relative path
            generated_creature.image_path = os.path.relpath(absolute_path, static_folder)     
            return render_template('index.html', color_to_sizes=color_to_sizes, color_to_limbs=color_to_limbs, color_to_heights=color_to_heights,
                           unique_sizes=unique_sizes,unique_limbs=unique_limbs,unique_heights=unique_heights,generated_creature=generated_creature,
                           selected_parents=selected_parents,parent_images=images_paths)   

    # Pass the unique attributes to the template
    return render_template('index.html', color_to_sizes=color_to_sizes, color_to_limbs=color_to_limbs, color_to_heights=color_to_heights,
                           unique_sizes=unique_sizes,unique_limbs=unique_limbs,unique_heights=unique_heights)        

def find_creature_by_attributes(attrs):
    base_image_path = "C:/Users/PC1/OneDrive/Documents/MonsterMash/static/web/static/images"

    for creature in creatures_list:
        # print(creature.color_code,creature.size,creature.limbs,creature.height)
        if (creature.color_code == attrs['color_code'] and
            creature.size == attrs['size'] and
            creature.limbs == attrs['limbs'] and
            creature.height == attrs['height']):
            creature_number=creatures_list.index(creature)+1

            image_path=f"{base_image_path}/Creature{creature_number}.jpg"
            return creature,image_path
    return None

def generate_offspring(parent1_attrs, parent2_attrs, environmental_factors, time_of_day):
    global creatures_list
    num_generations = 1
    mutation_rate = 0.01
    i=0;
    for generation in range(num_generations):
        new_generation = []

        while len(new_generation) < len(creatures_list):
            parent1 = select_parents(creatures_list, parent1_attrs['color_code'], parent1_attrs['size'], parent1_attrs['limbs'], parent1_attrs['height'])
            parent2 = select_parents(creatures_list, parent2_attrs['color_code'], parent2_attrs['size'], parent2_attrs['limbs'], parent2_attrs['height'])
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, mutation_rate)
            new_generation.append(offspring)
            save_path = f"C:/Users/PC1/OneDrive/Documents/MonsterMash/static/web/static/gen_images/Creature_gen{generation}_num{i}.jpg"
            save_array_as_image(offspring.image_array, save_path)
            offspring.image_path=save_path
            i+=1

    return new_generation[-1]

if __name__ == '__main__':
    app.run(debug=True)





