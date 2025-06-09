#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from graphviz import Digraph
from PIL import Image, ImageDraw, ImageFont
from syntactic_analyzer import SyntacticAnalyzer

class AutomatonVisualizer:
    """
    Classe para visualizar autômatos e gerar imagens/animações do caminho percorrido.
    """
    def __init__(self, output_dir='automaton_images'):
        """
        Inicializa o visualizador de autômatos.
        
        Args:
            output_dir: Diretório onde as imagens serão salvas
        """
        self.output_dir = output_dir
        
        # Criar diretório de saída se não existir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_automaton_graph(self, path=None, highlight_path=True):
        """
        Cria um grafo do autômato usando Graphviz.
        
        Args:
            path: Lista de tuplas (estado, símbolo) representando o caminho percorrido
            highlight_path: Se True, destaca o caminho percorrido
            
        Returns:
            Objeto Digraph do Graphviz
        """
        # Criar um novo grafo direcionado
        dot = Digraph(comment='Autômato Sintático')
        
        # Configurar o grafo
        dot.attr(rankdir='LR', size='8,5')
        dot.attr('node', shape='circle')
        
        # Definir os estados
        states = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']
        
        # Adicionar estados ao grafo
        for state in states:
            # Verificar se o estado está no caminho
            in_path = False
            if path:
                for s, _ in path:
                    if s == state:
                        in_path = True
                        break
            
            # Configurar o estilo do estado
            if state == 'q9':  # Estado final
                if in_path and highlight_path:
                    dot.node(state, style='filled', color='red', fillcolor='lightpink', shape='doublecircle')
                else:
                    dot.node(state, shape='doublecircle')
            else:  # Estados não-finais
                if in_path and highlight_path:
                    dot.node(state, style='filled', color='red', fillcolor='lightpink')
                else:
                    dot.node(state)
        
        # Definir as transições
        transitions = [
            ('q0', 'q1', 'DET_SUJEITO'),
            ('q0', 'q2', 'SUJEITO'),
            ('q1', 'q2', 'SUJEITO'),
            ('q2', 'q3', 'VERBO'),
            ('q3', 'q4', 'DET_OBJETO'),
            ('q3', 'q5', 'OBJETO'),
            ('q3', 'q5', 'PREDICATIVO'),
            ('q3', 'q5', 'VERBO_INFINITIVO'),
            ('q3', 'q5', 'ADVERBIO'),
            ('q3', 'q6', 'PREPOSICAO'),  # Adicionada transição direta de verbo para preposição
            ('q3', 'q9', 'PONTUACAO'),
            ('q4', 'q5', 'OBJETO'),
            ('q5', 'q6', 'PREPOSICAO'),
            ('q5', 'q5', 'ADVERBIO'),    # Advérbio após objeto ou outro advérbio
            ('q5', 'q9', 'PONTUACAO'),
            ('q6', 'q7', 'DET_OBJ_IND'),
            ('q6', 'q8', 'OBJ_INDIRETO'),
            ('q7', 'q8', 'OBJ_INDIRETO'),
            ('q8', 'q8', 'ADVERBIO'),    # Advérbio após objeto indireto
            ('q8', 'q9', 'PONTUACAO')
        ]
        
        # Adicionar transições ao grafo
        for src, dst, label in transitions:
            # Verificar se a transição está no caminho
            in_path = False
            if path:
                for i in range(1, len(path)):
                    prev_state, symbol = path[i-1][0], path[i][1]
                    curr_state = path[i][0]
                    if prev_state == src and curr_state == dst and symbol == label:
                        in_path = True
                        break
            
            # Configurar o estilo da transição
            if in_path and highlight_path:
                dot.edge(src, dst, label=label, color='red', penwidth='2.0')
            else:
                dot.edge(src, dst, label=label)
        
        return dot
    
    def generate_automaton_image(self, sentence, path, filename='automaton'):
        """
        Gera uma imagem PNG do autômato com o caminho destacado.
        
        Args:
            sentence: A frase analisada
            path: Lista de tuplas (estado, símbolo) representando o caminho percorrido
            filename: Nome do arquivo de saída (sem extensão)
            
        Returns:
            Caminho para a imagem gerada
        """
        # Criar o grafo
        dot = self.create_automaton_graph(path)
        
        # Adicionar título com a frase
        dot.attr(label=f'Autômato para a frase: "{sentence}"')
        
        # Renderizar o grafo como PNG
        output_path = os.path.join(self.output_dir, filename)
        dot.render(output_path, format='png', cleanup=True)
        
        return f"{output_path}.png"
    
    def generate_animation_frames(self, sentence, path, base_filename='frame'):
        """
        Gera frames para uma animação do caminho percorrido no autômato.
        
        Args:
            sentence: A frase analisada
            path: Lista de tuplas (estado, símbolo) representando o caminho percorrido
            base_filename: Prefixo para os nomes dos arquivos de frames
            
        Returns:
            Lista de caminhos para os frames gerados
        """
        frames = []
        
        # Gerar um frame para cada passo do caminho
        for i in range(1, len(path) + 1):
            # Criar o grafo com o caminho parcial
            partial_path = path[:i]
            dot = self.create_automaton_graph(partial_path)
            
            # Adicionar título com a frase e o passo atual
            if i > 1:
                current_token = path[i-1][1] if i-1 < len(path) and path[i-1][1] is not None else ""
                dot.attr(label=f'Autômato para a frase: "{sentence}"\nPasso {i-1}: Processando "{current_token}"')
            else:
                dot.attr(label=f'Autômato para a frase: "{sentence}"\nEstado inicial')
            
            # Renderizar o grafo como PNG
            frame_filename = f"{base_filename}_{i:03d}"
            output_path = os.path.join(self.output_dir, frame_filename)
            dot.render(output_path, format='png', cleanup=True)
            
            frames.append(f"{output_path}.png")
        
        return frames
    
    def create_gif_animation(self, frames, output_filename='animation.gif', duration=1000):
        """
        Cria uma animação GIF a partir dos frames.
        
        Args:
            frames: Lista de caminhos para os frames
            output_filename: Nome do arquivo GIF de saída
            duration: Duração de cada frame em milissegundos
            
        Returns:
            Caminho para o arquivo GIF
        """
        if not frames:
            return None
        
        # Carregar os frames como imagens PIL
        images = [Image.open(frame) for frame in frames]
        
        # Salvar como GIF
        output_path = os.path.join(self.output_dir, output_filename)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        
        return output_path
    
    def visualize_sentence(self, sentence):
        """
        Analisa uma frase e gera visualizações do autômato.
        
        Args:
            sentence: A frase a ser analisada
            
        Returns:
            Tupla (caminho_imagem, caminho_gif)
        """
        # Analisar a frase
        analyzer = SyntacticAnalyzer()
        is_valid, syntactic_tokens, path = analyzer.analyze_sentence(sentence)
        
        # Gerar imagem estática do autômato
        image_path = self.generate_automaton_image(sentence, path)
        
        # Gerar frames para animação
        frames = self.generate_animation_frames(sentence, path)
        
        # Criar animação GIF
        gif_path = self.create_gif_animation(frames)
        
        # Limpar os frames (opcional)
        for frame in frames:
            if os.path.exists(frame):
                os.remove(frame)
        
        return image_path, gif_path, is_valid, syntactic_tokens

def main():
    """Função principal para testar o visualizador."""
    if len(sys.argv) > 1:
        # Usar a frase fornecida como argumento
        sentence = ' '.join(sys.argv[1:])
    else:
        # Usar uma frase de exemplo
        sentence = "O professor explicou a matéria aos alunos."
    
    print(f"Analisando a frase: '{sentence}'")
    
    # Criar o visualizador
    visualizer = AutomatonVisualizer()
    
    # Gerar visualizações
    image_path, gif_path, is_valid, syntactic_tokens = visualizer.visualize_sentence(sentence)
    
    # Exibir resultados
    print("\nAnálise sintática:")
    for token, category, function in syntactic_tokens:
        print(f"  - '{token}': {category} → {function}")
    
    print(f"\nEstrutura sintática {'válida' if is_valid else 'inválida'}!")
    
    print(f"\nImagem do autômato salva em: {image_path}")
    print(f"Animação do autômato salva em: {gif_path}")
    
    # Abrir a imagem e o GIF (se disponível no sistema)
    try:
        os.system(f"xdg-open {image_path}")
        time.sleep(1)  # Pequeno atraso para não abrir ambos simultaneamente
        os.system(f"xdg-open {gif_path}")
    except:
        print("Não foi possível abrir as imagens automaticamente.")
        print("Por favor, abra-as manualmente usando um visualizador de imagens.")

if __name__ == "__main__":
    main()
