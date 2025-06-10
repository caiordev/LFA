#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from parser import SequentialStructureAnalyzer

def print_header():
    """Imprime o cabeçalho do programa."""
    print("\n" + "="*60)
    print("  VALIDADOR DE ESTRUTURA SEQUENCIAL USANDO AUTÔMATOS FINITOS")
    print("="*60)
    print("\nEste programa valida a estrutura sequencial de frases")
    print("em português brasileiro usando autômatos finitos.")
    print("\nO programa verifica se a ordem das classes gramaticais")
    print("(artigo, substantivo, verbo, etc.) segue padrões válidos.")
    print("\nExemplos de frases com estrutura válida:")
    print("  - O gato come o peixe.")
    print("  - Maria estuda matemática.")
    print("  - O homem inteligente lê o livro.")
    print("\nDigite 'sair' para encerrar o programa.")
    print("Digite 'exemplos' para ver mais exemplos.")
    print("="*60 + "\n")

def print_examples():
    """Imprime exemplos de frases válidas e inválidas."""
    print("\n" + "="*60)
    print("  EXEMPLOS DE FRASES")
    print("="*60)
    print("\nFrases com estrutura sequencial válida:")
    print("  - O gato come o peixe.")
    print("  - Maria estuda matemática.")
    print("  - O homem inteligente lê o livro.")
    print("  - A menina corre com o cachorro.")
    print("  - João compra um carro grande.")
    print("  - A professora explicou a matéria aos alunos.")
    
    print("\nFrases com estrutura sequencial inválida:")
    print("  - Gato o peixe come.")  # Ordem incorreta dos constituintes
    print("  - O come gato peixe.")  # Ordem incorreta dos constituintes
    print("  - O gato come.")        # Falta complemento verbal
    print("  - Estuda Maria.")       # Ordem incorreta (VS em vez de SV)
    print("="*60 + "\n")

def main():
    """Função principal do programa."""
    print_header()
    
    # Criar o analisador de estrutura sequencial
    analyzer = SequentialStructureAnalyzer()
    
    while True:
        # Solicitar entrada do usuário
        user_input = input("\nDigite uma frase para analisar (ou 'sair'/'exemplos'): ")
        
        # Verificar comandos especiais
        if user_input.lower() == 'sair':
            print("\nEncerrando o programa. Até logo!")
            break
        elif user_input.lower() == 'exemplos':
            print_examples()
            continue
        
        # Analisar a frase
        if user_input.strip():
            explanation = analyzer.explain_analysis(user_input)
            print("\n" + explanation)
        else:
            print("\nPor favor, digite uma frase válida.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário. Até logo!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nErro inesperado: {e}")
        sys.exit(1)
