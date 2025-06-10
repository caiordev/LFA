#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Este módulo define uma gramática simplificada para análise sintática de frases em português.
A gramática é representada como categorias gramaticais e regras de produção.
"""

# Categorias gramaticais
CATEGORIES = {
    'ARTIGO': ['o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas'],
    'PRONOME': [
        # Pronomes pessoais
        'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas', 'você', 'vocês',
        # Pronomes demonstrativos
        'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
        'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'isso', 'aquilo',
        # Pronomes possessivos
        'meu', 'minha', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas',
        'seu', 'sua', 'seus', 'suas', 'nosso', 'nossa', 'nossos', 'nossas'
    ],
    'SUBSTANTIVO': [
        # Pessoas
        'gato', 'cachorro', 'homem', 'mulher', 'menino', 'menina', 'professor', 'professora',
        'aluno', 'alunos', 'estudante', 'estudantes', 'maria', 'joão', 'pedro', 'ana',
        'diretor', 'diretora',
        # Objetos e conceitos
        'casa', 'carro', 'livro', 'computador', 'matemática', 'física', 'peixe', 'comida',
        'matéria', 'aula', 'escola', 'universidade', 'faculdade', 'curso', 'prova', 'trabalho',
        'exercício', 'projeto', 'apresentação', 'automato', 'autômato', 'sintaxe', 'gramática',
        'linguagem', 'programação', 'código', 'algoritmo', 'estrutura', 'validador'
    ],
    'VERBO': [
        # Verbos de ligação
        'sou', 'é', 'somos', 'são', 'era', 'eras', 'éramos', 'eram', 'foi', 'fomos',
        'foram', 'será', 'seremos', 'serão', 'seria', 'seriam', 'estou', 'está', 'estamos',
        'estão', 'estava', 'estávamos', 'estavam', 'estive', 'esteve', 'estivemos', 'estiveram',
        'ficar', 'fica', 'ficam', 'ficou', 'ficaram', 'parecer', 'parece', 'parecem', 'pareceu',
        'permanecer', 'permanece', 'permanecem', 'permaneceu', 'continuar', 'continua', 'continuam',
        # Presente
        'come', 'corre', 'pula', 'dorme', 'estuda', 'lê', 'escreve', 'fala', 'compra',
        'vende', 'ama', 'odeia', 'ensina', 'aprende', 'programa', 'desenvolve', 'cria',
        'implementa', 'valida', 'analisa', 'explica', 'caminha', 'liga',
        # Passado
        'comeu', 'correu', 'pulou', 'dormiu', 'estudou', 'leu', 'escreveu', 'falou',
        'comprou', 'vendeu', 'amou', 'odiou', 'ensinou', 'aprendeu', 'programou',
        'desenvolveu', 'criou', 'implementou', 'validou', 'analisou', 'explicou', 'caminhou',
        'liguei', 'ligou', 'ligaram',
        # Infinitivo
        'comer', 'correr', 'pular', 'dormir', 'estudar', 'ler', 'escrever', 'falar',
        'comprar', 'vender', 'amar', 'odiar', 'ensinar', 'aprender', 'programar',
        'desenvolver', 'criar', 'implementar', 'validar', 'analisar', 'explicar', 'caminhar',
        'ser', 'estar', 'ficar', 'parecer', 'permanecer', 'continuar', 'andar', 'viver', 'ligar'
    ],
    'ADJETIVO': [
        'bonito', 'bonita', 'grande', 'pequeno', 'pequena', 'alto', 'alta', 'baixo', 'baixa',
        'inteligente', 'esperto', 'esperta', 'rápido', 'rápida', 'lento', 'lenta',
        'difícil', 'fácil', 'complexo', 'complexa', 'simples', 'interessante', 'chato', 'chata',
        'importante', 'útil', 'inútil', 'necessário', 'necessária', 'eficiente',
        'feliz', 'triste', 'alegre', 'contente', 'satisfeito', 'satisfeita', 'animado', 'animada',
        'cansado', 'cansada', 'preocupado', 'preocupada', 'ocupado', 'ocupada', 'livre',
        'bom', 'boa', 'mau', 'má', 'melhor', 'pior', 'ótimo', 'ótima', 'péssimo', 'péssima',
        'novo', 'nova', 'velho', 'velha', 'jovem', 'antigo', 'antiga', 'moderno', 'moderna',
        'legal', 'bacana', 'maneiro', 'maneira', 'divertido', 'divertida', 'chique'
    ],
    'PREPOSICAO': [
        'de', 'para', 'com', 'sem', 'em', 'sobre', 'sob', 'entre', 'até', 'por',
        'ao', 'aos', 'à', 'às', 'do', 'dos', 'da', 'das', 'no', 'nos', 'na', 'nas'
    ],
    'PONTUACAO': ['.', '!', '?', ',', ';', ':'],
    'CONJUNCAO': ['e', 'ou', 'mas', 'porém', 'contudo', 'todavia', 'porque', 'pois', 'que', 'se'],
    'ADVERBIO': [
        'hoje', 'ontem', 'amanhã', 'agora', 'depois', 'antes', 'sempre', 'nunca',
        'rapidamente', 'lentamente', 'bem', 'mal', 'muito', 'pouco', 'bastante', 'demais',
        'aqui', 'ali', 'lá', 'acolá', 'longe', 'perto', 'dentro', 'fora', 'ainda'
    ]
}

# Lista de verbos de ligação
LINKING_VERBS = [
    'sou', 'é', 'somos', 'são', 'era', 'eras', 'éramos', 'eram', 'foi', 'fomos',
    'foram', 'será', 'seremos', 'serão', 'seria', 'seriam', 'estou', 'está', 'estamos',
    'estão', 'estava', 'estávamos', 'estavam', 'estive', 'esteve', 'estivemos', 'estiveram',
    'ficar', 'fica', 'ficam', 'ficou', 'ficaram', 'parecer', 'parece', 'parecem', 'pareceu',
    'permanecer', 'permanece', 'permanecem', 'permaneceu', 'continuar', 'continua', 'continuam',
    'andar', 'anda', 'andam', 'andou', 'andaram', 'viver', 'vive', 'vivem', 'viveu', 'viveram'
]

# Função para classificar uma palavra
def classify_word(word):
    """
    Classifica uma palavra de acordo com as categorias gramaticais definidas.
    Retorna a categoria da palavra ou None se não for encontrada.
    """
    word = word.lower()
    for category, words in CATEGORIES.items():
        if word in words:
            return category
    return None

# Função para verificar se um verbo é de ligação
def is_linking_verb(verb):
    """
    Verifica se um verbo é de ligação.
    Retorna True se for um verbo de ligação, False caso contrário.
    """
    return verb.lower() in LINKING_VERBS

# Estruturas sintáticas válidas
# Cada estrutura é uma sequência de categorias gramaticais que formam uma frase válida
VALID_STRUCTURES = [
    # Estruturas básicas
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['SUBSTANTIVO', 'VERBO', 'SUBSTANTIVO', 'PONTUACAO'],
    
    # Estruturas com pronomes
    ['PRONOME', 'VERBO', 'ADJETIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'ADJETIVO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['PRONOME', 'SUBSTANTIVO', 'VERBO', 'ADJETIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'PREPOSICAO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'ADVERBIO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'PREPOSICAO', 'SUBSTANTIVO', 'ADVERBIO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'ADVERBIO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'PREPOSICAO', 'SUBSTANTIVO', 'PONTUACAO', 'ADVERBIO', 'PONTUACAO'],
    
    # Estruturas com adjetivos
    ['ARTIGO', 'SUBSTANTIVO', 'ADJETIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'ADJETIVO', 'PONTUACAO'],
    ['ARTIGO', 'ADJETIVO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['SUBSTANTIVO', 'ADJETIVO', 'VERBO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['SUBSTANTIVO', 'VERBO', 'ADJETIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ADJETIVO', 'PONTUACAO'],
    
    # Estruturas com preposições
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PREPOSICAO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['SUBSTANTIVO', 'VERBO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    
    # Estrutura para frases como "A professora explicou a matéria aos alunos."
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PREPOSICAO', 'SUBSTANTIVO', 'PONTUACAO'],
    
    # Estruturas com conjunções
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'CONJUNCAO', 'VERBO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'CONJUNCAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
    ['PRONOME', 'VERBO', 'ADJETIVO', 'CONJUNCAO', 'ADJETIVO', 'PONTUACAO'],
    
    # Estruturas mais complexas
    ['ARTIGO', 'SUBSTANTIVO', 'ADJETIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'ADJETIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'VERBO', 'ARTIGO', 'SUBSTANTIVO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'ADJETIVO', 'PONTUACAO'],
    ['ARTIGO', 'SUBSTANTIVO', 'ADJETIVO', 'VERBO', 'PREPOSICAO', 'ARTIGO', 'SUBSTANTIVO', 'PONTUACAO'],
]
