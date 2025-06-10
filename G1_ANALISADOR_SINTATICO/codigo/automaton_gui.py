#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
from visualize_automaton import AutomatonVisualizer

class AutomatonGUI:
    """
    Interface gráfica para visualizar autômatos e analisar frases.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Autômatos Sintáticos")
        self.root.geometry("1000x700")
        
        # Configurar o estilo
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TLabel", font=("Arial", 12))
        self.style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        
        # Criar o visualizador de autômatos
        self.visualizer = AutomatonVisualizer()
        
        # Variáveis
        self.current_image_path = None
        self.current_gif_path = None
        self.image_label = None
        self.gif_label = None
        
        # Criar a interface
        self._create_widgets()
    
    def _create_widgets(self):
        """Cria os widgets da interface."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para entrada de texto
        input_frame = ttk.Frame(main_frame, padding=5)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Digite uma frase para análise:", style="Header.TLabel").pack(anchor=tk.W, pady=5)
        
        # Campo de entrada de texto
        self.sentence_entry = ttk.Entry(input_frame, font=("Arial", 12), width=70)
        self.sentence_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.sentence_entry.bind("<Return>", self._analyze_sentence)
        
        # Botão de análise
        analyze_button = ttk.Button(input_frame, text="Analisar", command=self._analyze_sentence)
        analyze_button.pack(side=tk.RIGHT, padx=5)
        
        # Frame para exemplos
        examples_frame = ttk.Frame(main_frame, padding=5)
        examples_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(examples_frame, text="Exemplos:").pack(side=tk.LEFT, padx=5)
        
        # Botões de exemplos
        example1 = ttk.Button(examples_frame, text="O gato come o peixe.", 
                             command=lambda: self._set_example("O gato come o peixe."))
        example1.pack(side=tk.LEFT, padx=5)
        
        example2 = ttk.Button(examples_frame, text="O professor explicou a matéria aos alunos.", 
                             command=lambda: self._set_example("O professor explicou a matéria aos alunos."))
        example2.pack(side=tk.LEFT, padx=5)
        
        example3 = ttk.Button(examples_frame, text="Maria estuda matemática.", 
                             command=lambda: self._set_example("Maria estuda matemática."))
        example3.pack(side=tk.LEFT, padx=5)
        
        # Frame para resultados
        results_frame = ttk.Frame(main_frame, padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Painel esquerdo para análise sintática
        left_panel = ttk.Frame(results_frame, padding=5, relief=tk.GROOVE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(left_panel, text="Análise Sintática:", style="Header.TLabel").pack(anchor=tk.W, pady=5)
        
        # Área de texto para mostrar a análise
        self.analysis_text = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, width=40, height=20, font=("Courier", 10))
        self.analysis_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Painel direito para visualizações
        right_panel = ttk.Frame(results_frame, padding=5, relief=tk.GROOVE, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(right_panel, text="Visualização do Autômato:", style="Header.TLabel").pack(anchor=tk.W, pady=5)
        
        # Frame para abas (imagem estática e animação)
        tab_control = ttk.Notebook(right_panel)
        tab_control.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Aba para imagem estática
        static_tab = ttk.Frame(tab_control)
        tab_control.add(static_tab, text="Imagem Estática")
        
        # Canvas para a imagem estática
        self.static_canvas = tk.Canvas(static_tab, bg="white")
        self.static_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Aba para animação
        animation_tab = ttk.Frame(tab_control)
        tab_control.add(animation_tab, text="Animação")
        
        # Canvas para a animação
        self.animation_canvas = tk.Canvas(animation_tab, bg="white")
        self.animation_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Botões para salvar imagens
        buttons_frame = ttk.Frame(right_panel, padding=5)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        save_static_button = ttk.Button(buttons_frame, text="Abrir Imagem", command=self._open_static_image)
        save_static_button.pack(side=tk.LEFT, padx=5)
        
        save_animation_button = ttk.Button(buttons_frame, text="Abrir Animação", command=self._open_animation)
        save_animation_button.pack(side=tk.LEFT, padx=5)
        
        # Mensagem inicial
        self.analysis_text.insert(tk.END, "Digite uma frase e clique em 'Analisar' para começar.\n\n")
        self.analysis_text.insert(tk.END, "O autômato irá analisar a estrutura sintática da frase, identificando funções como sujeito, verbo, objeto direto e objeto indireto.\n\n")
        self.analysis_text.insert(tk.END, "Você também pode clicar em um dos exemplos acima.")
        self.analysis_text.config(state=tk.DISABLED)
    
    def _set_example(self, example):
        """Define um exemplo no campo de entrada."""
        self.sentence_entry.delete(0, tk.END)
        self.sentence_entry.insert(0, example)
        self._analyze_sentence()
    
    def _analyze_sentence(self, event=None):
        """Analisa a frase e atualiza a interface."""
        sentence = self.sentence_entry.get().strip()
        
        if not sentence:
            messagebox.showwarning("Aviso", "Por favor, digite uma frase para analisar.")
            return
        
        try:
            # Analisar a frase e gerar visualizações
            image_path, gif_path, is_valid, syntactic_tokens = self.visualizer.visualize_sentence(sentence)
            
            # Atualizar os caminhos das imagens
            self.current_image_path = image_path
            self.current_gif_path = gif_path
            
            # Atualizar a área de texto com a análise
            self.analysis_text.config(state=tk.NORMAL)
            self.analysis_text.delete(1.0, tk.END)
            
            self.analysis_text.insert(tk.END, f"Frase: {sentence}\n\n")
            self.analysis_text.insert(tk.END, "Análise Sintática:\n")
            
            for token, category, function in syntactic_tokens:
                self.analysis_text.insert(tk.END, f"  - '{token}': {category} → {function}\n")
            
            self.analysis_text.insert(tk.END, f"\nResultado da Análise:\n")
            if is_valid:
                self.analysis_text.insert(tk.END, "  ✓ Estrutura sintática válida!\n")
                
                # Identificar as principais partes da oração
                sujeito = [token for token, _, function in syntactic_tokens 
                          if function in ["SUJEITO", "DET_SUJEITO", "MODIFICADOR_SUJEITO"]]
                predicado = [token for token, _, function in syntactic_tokens 
                            if function not in ["SUJEITO", "DET_SUJEITO", "MODIFICADOR_SUJEITO", "PONTUACAO"]]
                
                self.analysis_text.insert(tk.END, "\nPartes da Oração:\n")
                self.analysis_text.insert(tk.END, f"  - Sujeito: {' '.join(sujeito)}\n")
                self.analysis_text.insert(tk.END, f"  - Predicado: {' '.join(predicado)}\n")
                
                # Identificar componentes específicos
                verbo = [token for token, _, function in syntactic_tokens if function == "VERBO"]
                objeto_direto = [token for token, _, function in syntactic_tokens 
                               if function in ["OBJETO", "DET_OBJETO", "MODIFICADOR_OBJETO"]]
                objeto_indireto = [token for token, _, function in syntactic_tokens 
                                 if function in ["OBJ_INDIRETO", "DET_OBJ_IND", "MODIFICADOR_OBJ_IND", "PREPOSICAO"]]
                adverbios = [token for token, _, function in syntactic_tokens if function == "ADVERBIO"]
                
                if verbo:
                    self.analysis_text.insert(tk.END, f"  - Verbo: {' '.join(verbo)}\n")
                if objeto_direto:
                    self.analysis_text.insert(tk.END, f"  - Objeto Direto: {' '.join(objeto_direto)}\n")
                if objeto_indireto:
                    self.analysis_text.insert(tk.END, f"  - Objeto Indireto: {' '.join(objeto_indireto)}\n")
                if adverbios:
                    self.analysis_text.insert(tk.END, f"  - Advérbio: {' '.join(adverbios)}\n")
            else:
                self.analysis_text.insert(tk.END, "  ✗ Estrutura sintática inválida!\n")
                self.analysis_text.insert(tk.END, "  A frase não segue um padrão sintático reconhecido.\n")
            
            self.analysis_text.config(state=tk.DISABLED)
            
            # Atualizar as imagens
            self._update_images(image_path, gif_path)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao analisar a frase: {str(e)}")
    
    def _update_images(self, image_path, gif_path):
        """Atualiza as imagens na interface."""
        # Limpar os canvas
        self.static_canvas.delete("all")
        self.animation_canvas.delete("all")
        
        # Carregar e exibir a imagem estática
        if os.path.exists(image_path):
            img = Image.open(image_path)
            # Redimensionar a imagem para caber no canvas
            canvas_width = self.static_canvas.winfo_width()
            canvas_height = self.static_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Verificar se o canvas já tem dimensões
                img = self._resize_image(img, canvas_width, canvas_height)
                
                self.static_image = ImageTk.PhotoImage(img)
                self.static_canvas.create_image(canvas_width//2, canvas_height//2, image=self.static_image, anchor=tk.CENTER)
        
        # Carregar e exibir a animação
        if os.path.exists(gif_path):
            # Para GIFs animados, precisamos usar um método diferente
            try:
                from PIL import ImageSequence
                
                # Abrir o GIF
                gif = Image.open(gif_path)
                
                # Redimensionar o GIF
                canvas_width = self.animation_canvas.winfo_width()
                canvas_height = self.animation_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Criar uma lista de frames redimensionados
                    frames = []
                    for frame in ImageSequence.Iterator(gif):
                        resized_frame = self._resize_image(frame.copy(), canvas_width, canvas_height)
                        frames.append(ImageTk.PhotoImage(resized_frame))
                    
                    # Exibir o primeiro frame
                    if frames:
                        self.animation_frames = frames
                        self.current_frame = 0
                        self.animation_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                          image=self.animation_frames[0], anchor=tk.CENTER, tags="gif")
                        
                        # Iniciar a animação
                        self._animate_gif(0)
            except Exception as e:
                print(f"Erro ao carregar o GIF: {str(e)}")
    
    def _resize_image(self, img, canvas_width, canvas_height):
        """Redimensiona uma imagem para caber no canvas."""
        # Calcular a proporção da imagem
        img_width, img_height = img.size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        
        # Usar a menor proporção para manter o aspecto da imagem
        ratio = min(width_ratio, height_ratio) * 0.9  # 90% do tamanho para deixar uma margem
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def _animate_gif(self, frame_index):
        """Anima o GIF frame a frame."""
        if not hasattr(self, 'animation_frames') or not self.animation_frames:
            return
        
        # Atualizar a imagem
        self.animation_canvas.delete("gif")
        canvas_width = self.animation_canvas.winfo_width()
        canvas_height = self.animation_canvas.winfo_height()
        self.animation_canvas.create_image(canvas_width//2, canvas_height//2, 
                                          image=self.animation_frames[frame_index], anchor=tk.CENTER, tags="gif")
        
        # Avançar para o próximo frame
        next_frame = (frame_index + 1) % len(self.animation_frames)
        
        # Agendar a próxima atualização (500ms por frame)
        self.root.after(500, self._animate_gif, next_frame)
    
    def _open_static_image(self):
        """Abre a imagem estática em um visualizador externo."""
        if self.current_image_path and os.path.exists(self.current_image_path):
            try:
                os.system(f"xdg-open {self.current_image_path}")
            except:
                messagebox.showinfo("Informação", f"A imagem está salva em:\n{self.current_image_path}")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem disponível. Analise uma frase primeiro.")
    
    def _open_animation(self):
        """Abre a animação em um visualizador externo."""
        if self.current_gif_path and os.path.exists(self.current_gif_path):
            try:
                os.system(f"xdg-open {self.current_gif_path}")
            except:
                messagebox.showinfo("Informação", f"A animação está salva em:\n{self.current_gif_path}")
        else:
            messagebox.showwarning("Aviso", "Nenhuma animação disponível. Analise uma frase primeiro.")

def main():
    """Função principal para iniciar a interface gráfica."""
    root = tk.Tk()
    app = AutomatonGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
