import numpy as np
from PIL import Image, ImageDraw
import random
import time

# Função para calcular o Erro Quadrático Médio (MSE)
def calculate_mse(img1, img2):
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse

# Função para gerar uma imagem a partir de um cromossomo
def chromosome_to_image(chromosome, img_size=(64, 64)):
    img = Image.new('RGB', img_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Interpreta cada gene (blocos de cor)
    for i in range(0, len(chromosome), 5):
        x = chromosome[i] % img_size[0]
        y = chromosome[i + 1] % img_size[1]
        size = chromosome[i + 2] % 20  # Tamanho do bloco
        color = (chromosome[i + 3] % 256, chromosome[i + 4] % 256, (chromosome[i + 3] + chromosome[i + 4]) % 256)
        draw.rectangle([x, y, x + size, y + size], fill=color)

    return img

# Função de criação da população inicial
def create_population(pop_size, chromosome_length):
    return [np.random.randint(0, 256, chromosome_length).tolist() for _ in range(pop_size)]

# Função de crossover
def crossover(parent1, parent2):
    split_point = random.randint(0, len(parent1) - 1)
    child = parent1[:split_point] + parent2[split_point:]
    return child

# Função de mutação
def mutate(chromosome, mutation_rate=0.05):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, 255)
    return chromosome

# Função de seleção
def select_population(population, fitness_scores, num_parents):
    selected_parents = random.choices(population, weights=fitness_scores, k=num_parents)
    return selected_parents

# Função principal do Algoritmo Genético
def genetic_algorithm(original_img, population_size=100, chromosome_length=150, generations=1000, mutation_rate=0.05, mse_threshold=100, time_limit=60):
    start_time = time.time()

    # Carregar a imagem original e redimensionar para 64x64
    original_img = original_img.resize((64, 64))

    # Criar a população inicial
    population = create_population(population_size, chromosome_length)

    for generation in range(generations):
        # Avaliar a população (fitness)
        fitness_scores = []
        for chrom in population:
            generated_img = chromosome_to_image(chrom)
            mse = calculate_mse(original_img, generated_img)
            fitness_scores.append(1 / (1 + mse))  # O menor MSE terá a melhor fitness

        # Verificar se algum cromossomo atinge o critério de MSE
        best_mse = 1 / max(fitness_scores) - 1
        if best_mse < mse_threshold:
            best_chromosome = population[np.argmax(fitness_scores)]
            print(f"Gerado com sucesso na geração {generation} com MSE: {best_mse}")
            return chromosome_to_image(best_chromosome)

        # Verificar o tempo limite
        if time.time() - start_time > time_limit:
            best_chromosome = population[np.argmax(fitness_scores)]
            print(f"Tempo limite atingido. Melhor MSE: {best_mse}")
            return chromosome_to_image(best_chromosome)

        # Seleção dos pais
        num_parents = population_size // 2
        parents = select_population(population, fitness_scores, num_parents)

        # Gerar nova população
        new_population = []
        for _ in range(population_size // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population

    best_chromosome = population[np.argmax(fitness_scores)]
    print(f"Finalizado com MSE: {best_mse}")
    return chromosome_to_image(best_chromosome)

# Carregar a imagem original e executar o algoritmo
original_img = Image.open("imagem_original.jpg")
resultado = genetic_algorithm(original_img)

# Salvar o resultado
resultado.save("imagem_gerada.jpg")
print("Imagem gerada salva como 'imagem_gerada.jpg'")
