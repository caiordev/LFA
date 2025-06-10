#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from syntactic_analyzer import SyntacticAnalyzer

def print_header():
    """Imprime o cabeçalho do programa."""
    print("\n" + "="*70)
    print("  ANALISADOR SINTÁTICO COMPLETO USANDO AUTÔMATOS FINITOS")
    print("="*70)
    print("\nEste programa realiza análise sintática de frases em português brasileiro,")
    print("identificando funções sintáticas como sujeito, verbo, objeto direto e indireto.")
    print("\nExemplos de frases para análise:")
    print("  - O gato come o peixe.")
    print("  - Maria estuda matemática.")
    print("  - O professor explicou a matéria aos alunos.")
    print("\nDigite 'sair' para encerrar o programa.")
    print("Digite 'exemplos' para ver mais exemplos.")
    print("Digite 'automato' para ver a descrição do autômato utilizado.")
    print("="*70 + "\n")

def print_examples():
    """Imprime exemplos de frases para análise sintática."""
    print("\n" + "="*70)
    print("  EXEMPLOS DE FRASES PARA ANÁLISE SINTÁTICA")
    print("="*70)
    print("\nFrases simples (sujeito + verbo + objeto):")
    print("  - O gato come o peixe.")
    print("  - Maria estuda matemática.")
    print("  - Os alunos aprendem programação.")
    
    print("\nFrases com objeto indireto:")
    print("  - O professor explicou a matéria aos alunos.")
    print("  - Maria deu o livro para João.")
    print("  - O homem vendeu o carro ao vizinho.")
    
    print("\nFrases que o analisador pode não processar corretamente:")
    print("  - Frases na voz passiva: 'O livro foi lido por Maria.'")
    print("  - Frases com orações subordinadas: 'O homem que vimos ontem chegou.'")
    print("  - Frases com estruturas complexas: 'Embora estivesse chovendo, ele saiu.'")
    print("="*70 + "\n")

def main():
    """Função principal do programa."""
    print_header()
    
    # Criar o analisador sintático
    analyzer = SyntacticAnalyzer()
    
    while True:
        # Solicitar entrada do usuário
        user_input = input("\nDigite uma frase para análise sintática (ou 'sair'/'exemplos'/'automato'): ")
        
        # Verificar comandos especiais
        if user_input.lower() == 'sair':
            print("\nEncerrando o programa. Até logo!")
            break
        elif user_input.lower() == 'exemplos':
            print_examples()
            continue
        elif user_input.lower() == 'automato':
            print("\n" + analyzer.get_automaton_description())
            continue
        
        # Analisar a frase
        if user_input.strip():
            # Obter a explicação da análise e a visualização do autômato
            explanation, automaton_visualization = analyzer.explain_analysis(user_input)
            print("\n" + explanation)
            
            # Mostrar o autômato com o caminho percorrido
            print("\n" + automaton_visualization)
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
