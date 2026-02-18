import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ==================== БАЗОВЫЕ КЛАССЫ ====================

@dataclass
class UniversalConstants:
    """Базовые константы нашей Вселенной"""
    hbar: float = 1.0545718e-34  # Дж·с
    c: float = 299792458.0  # м/с
    epsilon_0: float = 8.8541878128e-12  # Ф/м
    G: float = 6.67430e-11  # м^3·кг^-1·с^-2
    m_e: float = 9.10938356e-31  # кг
    m_p: float = 1.6726219e-27  # кг
    m_n: float = 1.674927471e-27  # кг
    k_B: float = 1.380649e-23  # Дж/К
    e: float = 1.60217662e-19  # Кл

class UniverseParameters:
    """Полное описание Вселенной"""
    
    def __init__(self, name="Our Universe", alpha=None, e=None, m_p=None, 
                 hbar=None, c=None, G=None, epsilon_0=None):
        self.name = name
        self.const = UniversalConstants()
        
        self.hbar = hbar if hbar else self.const.hbar
        self.c = c if c else self.const.c
        self.G = G if G else self.const.G
        self.epsilon_0 = epsilon_0 if epsilon_0 else self.const.epsilon_0
        
        if alpha is not None:
            self.alpha = alpha
            self.e = math.sqrt(alpha * 4 * math.pi * self.epsilon_0 * self.hbar * self.c)
        elif e is not None:
            self.e = e
            self.alpha = (e**2) / (4 * math.pi * self.epsilon_0 * self.hbar * self.c)
        else:
            self.e = self.const.e
            self.alpha = (self.e**2) / (4 * math.pi * self.epsilon_0 * self.hbar * self.c)
        
        self.m_p = m_p if m_p else self.const.m_p
        
        # Планковские единицы
        self.m_planck = math.sqrt(self.hbar * self.c / self.G)
        self.l_planck = math.sqrt(self.hbar * self.G / self.c**3)
        self.t_planck = self.l_planck / self.c
        self.q_planck = math.sqrt(4 * math.pi * self.epsilon_0 * self.hbar * self.c)
    
    def __repr__(self):
        return f"{self.name}: α={self.alpha:.6f}, e/e₀={self.e/self.const.e:.3f}"

# ==================== ФИЗИЧЕСКИЕ МОДУЛИ ====================

class AtomicPhysics:
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def bohr_radius(self) -> float:
        return (4 * math.pi * self.u.epsilon_0 * self.u.hbar**2) / (self.u.const.m_e * self.u.e**2)
    
    def rydberg_energy(self) -> float:
        return (self.u.const.m_e * self.u.e**4) / (32 * math.pi**2 * self.u.epsilon_0**2 * self.u.hbar**2)
    
    def rydberg_ev(self) -> float:
        return self.rydberg_energy() / self.u.const.e
    
    def compton_wavelength(self) -> float:
        return self.u.hbar / (self.u.const.m_e * self.u.c)
    
    def fine_structure_effects(self) -> Dict[str, float]:
        """Все атомные эффекты, связанные с α"""
        a0 = self.bohr_radius()
        λ_c = self.compton_wavelength()
        return {
            'a0': a0,
            'a0_norm': a0 / 5.29e-11,  # относительно нашей Вселенной
            'a0_over_λc': a0 / λ_c,
            'E_bind': self.rydberg_ev(),
            'E_bind_norm': self.rydberg_ev() / 13.6  # относительно нашей Вселенной
        }

class NuclearPhysics:
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def qcd_scale(self, alpha_dependence: float = 0.1) -> float:
        """Масштаб КХД с возможностью регулировать зависимость от α"""
        base_lambda = 2.5e-28  # кг
        alpha_ratio = self.u.alpha / (1/137.036)
        # Логарифмическая зависимость от α (теоретически обоснованная)
        correction = 1 + alpha_dependence * math.log(alpha_ratio)
        return base_lambda * max(0.3, min(3.0, correction))
    
    def binding_energy(self, A: int = 56, alpha_dependence: float = 0.1) -> float:
        """Энергия связи ядра с учетом зависимости от α"""
        e0 = 8.5e6 * self.u.const.e
        
        Z = A/2
        # Кулоновская часть растет с α
        coulomb = (self.u.alpha / (1/137.036)) * (Z**2) / (A**(4/3))
        # Сильная часть зависит от масштаба КХД
        strong = self.qcd_scale(alpha_dependence) / 2.5e-28
        
        binding = e0 * (strong - 0.1 * coulomb)
        return max(0, binding)
    
    def coulomb_barrier(self, Z1: int, Z2: int) -> float:
        r = 1.2e-15 * ((Z1 + Z2) ** (1/3))
        return (Z1 * Z2 * self.u.alpha * self.u.hbar * self.u.c) / (4 * math.pi * r)

class StellarPhysics:
    def __init__(self, universe: UniverseParameters, nuclear: NuclearPhysics):
        self.u = universe
        self.nuclear = nuclear
        self._pp_rate_ref = None
        
    def pp_chain_rate(self, T: float = 1.5e7) -> float:
        """Скорость pp-цепи при заданной температуре"""
        kT = self.u.const.k_B * T
        m_reduced = self.u.const.m_p / 2
        
        # Гамов-фактор (туннелирование)
        E_G = (math.pi * self.u.alpha)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        # Скорость реакции
        rate = self.u.alpha**2 * (kT)**(2/3) * gamow
        
        if self._pp_rate_ref is None:
            self._pp_rate_ref = rate
            
        return rate / self._pp_rate_ref
    
    def triple_alpha(self, T: float = 1e8) -> Tuple[float, float]:
        """Анализ тройной альфа-реакции"""
        E_res = 380e3 * self.u.const.e  # энергия резонанса в нашей Вселенной
        kT = self.u.const.k_B * T
        
        # Сдвиг резонанса с α (квадратичная зависимость)
        E_res_actual = E_res * (self.u.alpha / (1/137.036))**2
        
        # Насколько близко резонанс к тепловой энергии
        resonance_match = math.exp(-abs(E_res_actual - 3*kT) / kT)
        
        return E_res_actual / self.u.const.e / 1000, resonance_match
    
    def cno_cycle_rate(self, T: float = 3e7) -> float:
        """Скорость CNO-цикла"""
        kT = self.u.const.k_B * T
        Z_avg = 7  # средний Z для C,N,O
        
        E_G = (math.pi * self.u.alpha * Z_avg)**2 * (self.u.const.m_p * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        return self.u.alpha * gamow

class HabitabilityIndex(Enum):
    """Индекс пригодности для жизни"""
    DEAD = 0
    HOSTILE = 1
    MARGINAL = 2
    HABITABLE = 3
    OPTIMAL = 4

class UniverseAnalyzer:
    """Полный анализ Вселенной с вычислением индекса пригодности"""
    
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        self.atomic = AtomicPhysics(universe)
        self.nuclear = NuclearPhysics(universe)
        self.stellar = StellarPhysics(universe, self.nuclear)
        
    def calculate_habitability_index(self) -> Tuple[HabitabilityIndex, float, Dict]:
        """Вычисляет индекс пригодности для жизни"""
        
        # Собираем все метрики
        metrics = {}
        
        # 1. Атомная структура
        atomic_effects = self.atomic.fine_structure_effects()
        a0_ratio = atomic_effects['a0_over_λc']
        
        if 10 < a0_ratio < 1000:
            metrics['atomic'] = 1.0
        elif 1 < a0_ratio < 10000:
            metrics['atomic'] = 0.5
        else:
            metrics['atomic'] = 0.0
        
        # 2. Химия (значение α)
        α = self.u.alpha
        if 1/200 < α < 1/50:  # оптимальный диапазон
            metrics['chemistry'] = 1.0
        elif 1/300 < α < 1/30:  # допустимый диапазон
            metrics['chemistry'] = 0.5
        else:
            metrics['chemistry'] = 0.0
        
        # 3. Ядерная энергия
        binding = self.nuclear.binding_energy() / (self.u.const.e * 1e6)  # МэВ
        if 5 < binding < 12:
            metrics['nuclear'] = 1.0
        elif 2 < binding < 15:
            metrics['nuclear'] = 0.5
        else:
            metrics['nuclear'] = 0.0
        
        # 4. Звездный синтез (углерод)
        _, res_match = self.stellar.triple_alpha()
        if res_match > 0.5:
            metrics['carbon'] = 1.0
        elif res_match > 0.1:
            metrics['carbon'] = 0.5
        else:
            metrics['carbon'] = 0.0
        
        # 5. Звездный синтез (водород)
        pp_rate = self.stellar.pp_chain_rate()
        if 0.1 < pp_rate < 10:
            metrics['fusion'] = 1.0
        elif 0.01 < pp_rate < 100:
            metrics['fusion'] = 0.5
        else:
            metrics['fusion'] = 0.0
        
        # Вычисляем общий индекс (среднее взвешенное)
        weights = {'atomic': 0.2, 'chemistry': 0.3, 'nuclear': 0.2, 'carbon': 0.2, 'fusion': 0.1}
        total_score = sum(metrics[k] * weights[k] for k in metrics)
        
        # Определяем категорию
        if total_score > 0.8:
            index = HabitabilityIndex.OPTIMAL
        elif total_score > 0.6:
            index = HabitabilityIndex.HABITABLE
        elif total_score > 0.3:
            index = HabitabilityIndex.MARGINAL
        elif total_score > 0.1:
            index = HabitabilityIndex.HOSTILE
        else:
            index = HabitabilityIndex.DEAD
            
        return index, total_score, metrics
    
    def get_all_properties(self) -> Dict:
        """Возвращает все свойства Вселенной"""
        atomic = self.atomic.fine_structure_effects()
        
        return {
            'alpha': self.u.alpha,
            'e_ratio': self.u.e / self.u.const.e,
            'bohr_radius_norm': atomic['a0_norm'],
            'binding_energy_mev': self.nuclear.binding_energy() / (self.u.const.e * 1e6),
            'pp_rate': self.stellar.pp_chain_rate(),
            'triple_alpha_res_match': self.stellar.triple_alpha()[1],
            'cno_rate': self.stellar.cno_cycle_rate()
        }

# ==================== ДИНАМИЧЕСКОЕ ИССЛЕДОВАНИЕ ====================

class MultiverseDynamicsExplorer:
    """Исследователь динамики мультивселенной"""
    
    def __init__(self, base_universe: Optional[UniverseParameters] = None):
        self.base = base_universe if base_universe else UniverseParameters("Base")
        self.results = {}
        
    def scan_parameter(self, param_name: str, 
                       start: float, stop: float, 
                       num_points: int = 100,
                       log_scale: bool = False,
                       other_params: Optional[Dict] = None) -> Dict:
        """
        Сканирует один параметр и собирает все зависимости
        
        Args:
            param_name: "alpha", "e", или "m_p"
            start, stop: границы сканирования
            num_points: количество точек
            log_scale: использовать логарифмическую шкалу
            other_params: фиксированные значения других параметров
        """
        
        print(f"\n🔍 Сканирование параметра {param_name} от {start} до {stop} ({num_points} точек)...")
        
        if log_scale:
            values = np.logspace(np.log10(start), np.log10(stop), num_points)
        else:
            values = np.linspace(start, stop, num_points)
        
        param_values = []
        properties_list = []
        indices = []
        scores = []
        
        other_params = other_params or {}
        
        for i, val in enumerate(values):
            # Создаем вселенную с текущим значением параметра
            if param_name == "alpha":
                u = UniverseParameters(
                    name=f"α={val:.6f}",
                    alpha=val,
                    **{k: v for k, v in other_params.items() if k != 'alpha'}
                )
                param_values.append(val)
            elif param_name == "e":
                e_val = val * self.base.const.e
                u = UniverseParameters(
                    name=f"e/e₀={val:.3f}",
                    e=e_val,
                    **{k: v for k, v in other_params.items() if k != 'e'}
                )
                param_values.append(val)  # сохраняем относительное значение
            elif param_name == "m_p":
                m_p_val = val * self.base.const.m_p
                u = UniverseParameters(
                    name=f"m_p/m_p₀={val:.3f}",
                    m_p=m_p_val,
                    **{k: v for k, v in other_params.items() if k != 'm_p'}
                )
                param_values.append(val)
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            # Анализируем вселенную
            analyzer = UniverseAnalyzer(u)
            props = analyzer.get_all_properties()
            index, score, metrics = analyzer.calculate_habitability_index()
            
            properties_list.append(props)
            indices.append(index.value)
            scores.append(score)
            
            if i % max(1, num_points//10) == 0:
                print(f"   Прогресс: {i}/{num_points} ({i/num_points*100:.1f}%)")
        
        # Сохраняем результаты
        result = {
            'param_name': param_name,
            'param_values': np.array(param_values),
            'properties': properties_list,
            'habitability_indices': np.array(indices),
            'habitability_scores': np.array(scores)
        }
        
        self.results[param_name] = result
        print(f"✅ Сканирование завершено!")
        
        return result
    
    def scan_2d(self, param1_name: str, param1_range: Tuple[float, float],
                param2_name: str, param2_range: Tuple[float, float],
                num_points1: int = 30, num_points2: int = 30) -> Dict:
        """
        Двумерное сканирование для поиска корреляций
        """
        print(f"\n🔬 2D сканирование: {param1_name} × {param2_name}")
        
        if param1_name == "alpha":
            values1 = np.linspace(param1_range[0], param1_range[1], num_points1)
        else:
            values1 = np.linspace(param1_range[0], param1_range[1], num_points1)
            
        if param2_name == "alpha":
            values2 = np.linspace(param2_range[0], param2_range[1], num_points2)
        else:
            values2 = np.linspace(param2_range[0], param2_range[1], num_points2)
        
        score_map = np.zeros((num_points1, num_points2))
        
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                params = {}
                
                if param1_name == "alpha":
                    params['alpha'] = v1
                elif param1_name == "e":
                    params['e'] = v1 * self.base.const.e
                elif param1_name == "m_p":
                    params['m_p'] = v1 * self.base.const.m_p
                
                if param2_name == "alpha":
                    params['alpha'] = v2
                elif param2_name == "e":
                    params['e'] = v2 * self.base.const.e
                elif param2_name == "m_p":
                    params['m_p'] = v2 * self.base.const.m_p
                
                u = UniverseParameters(name=f"2D-{i}-{j}", **params)
                analyzer = UniverseAnalyzer(u)
                _, score, _ = analyzer.calculate_habitability_index()
                score_map[i, j] = score
        
        result = {
            'param1': param1_name,
            'param2': param2_name,
            'values1': values1,
            'values2': values2,
            'score_map': score_map
        }
        
        return result
    
    def find_critical_points(self, param_name: str, threshold: float = 0.5) -> List[float]:
        """Находит критические значения параметра, где пригодность падает ниже порога"""
        if param_name not in self.results:
            raise ValueError(f"Сначала выполните сканирование для {param_name}")
        
        result = self.results[param_name]
        scores = result['habitability_scores']
        values = result['param_values']
        
        critical_points = []
        for i in range(len(scores)-1):
            if (scores[i] - threshold) * (scores[i+1] - threshold) < 0:
                # Интерполяция
                t = (threshold - scores[i]) / (scores[i+1] - scores[i])
                crit_val = values[i] + t * (values[i+1] - values[i])
                critical_points.append(crit_val)
        
        return critical_points
    
    def analyze_correlations(self, param_name: str) -> Dict:
        """Анализирует корреляции между параметром и различными свойствами"""
        if param_name not in self.results:
            raise ValueError(f"Сначала выполните сканирование для {param_name}")
        
        result = self.results[param_name]
        values = result['param_values']
        props = result['properties']
        
        # Извлекаем все ключи свойств
        if not props:
            return {}
        
        keys = props[0].keys()
        correlations = {}
        
        for key in keys:
            prop_values = [p[key] for p in props]
            
            # Вычисляем корреляцию Пирсона
            corr = np.corrcoef(values, prop_values)[0, 1]
            correlations[key] = corr
        
        return correlations

# ==================== ВИЗУАЛИЗАЦИЯ ====================

class MultiverseVisualizer:
    """Визуализация результатов исследования мультивселенной"""
    
    def __init__(self, explorer: MultiverseDynamicsExplorer):
        self.explorer = explorer
        
    def plot_1d_scan(self, param_name: str, 
                     properties: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (15, 10)):
        """Строит графики 1D сканирования"""
        
        if param_name not in self.explorer.results:
            print(f"Нет данных для {param_name}")
            return
        
        result = self.explorer.results[param_name]
        values = result['param_values']
        
        if properties is None:
            # Автоматически выбираем свойства для отображения
            properties = ['bohr_radius_norm', 'binding_energy_mev', 
                         'pp_rate', 'triple_alpha_res_match']
        
        n_props = len(properties)
        fig, axes = plt.subplots(n_props, 2, figsize=figsize)
        
        # Цветовая карта для индекса пригодности
        colors = plt.cm.RdYlGn(result['habitability_scores'])
        
        for i, prop in enumerate(properties):
            # Левый график: зависимость свойства
            ax = axes[i, 0] if n_props > 1 else axes[0]
            prop_values = [p[prop] for p in result['properties']]
            
            ax.scatter(values, prop_values, c=colors, alpha=0.6, s=30)
            ax.set_xlabel(param_name)
            ax.set_ylabel(prop)
            ax.set_title(f'{prop} vs {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Добавляем нашу Вселенную для reference
            our_val = 1/137.036 if param_name == 'alpha' else 1.0
            if our_val >= min(values) and our_val <= max(values):
                ax.axvline(x=our_val, color='red', linestyle='--', alpha=0.5, label='Наша Вселенная')
            
            # Правый график: индекс пригодности
            ax = axes[i, 1] if n_props > 1 else axes[1]
            ax.scatter(values, prop_values, c=result['habitability_scores'], 
                      cmap='RdYlGn', vmin=0, vmax=1, s=30)
            ax.set_xlabel(param_name)
            ax.set_ylabel(prop)
            ax.set_title(f'{prop} (цвет = пригодность)')
            ax.grid(True, alpha=0.3)
            
            if our_val >= min(values) and our_val <= max(values):
                ax.axvline(x=our_val, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_habitability_scan(self, param_name: str, figsize: Tuple[int, int] = (12, 5)):
        """Строит график пригодности в зависимости от параметра"""
        
        result = self.explorer.results[param_name]
        values = result['param_values']
        scores = result['habitability_scores']
        indices = result['habitability_indices']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # График 1: непрерывная шкала пригодности
        ax1.plot(values, scores, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(values, 0, scores, alpha=0.3, color='green')
        ax1.axhline(y=0.8, color='gold', linestyle='--', alpha=0.5, label='Оптимально')
        ax1.axhline(y=0.6, color='lime', linestyle='--', alpha=0.5, label='Пригодно')
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Маргинально')
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Враждебно')
        
        # Наша Вселенная
        our_val = 1/137.036 if param_name == 'alpha' else 1.0
        if our_val >= min(values) and our_val <= max(values):
            ax1.axvline(x=our_val, color='red', linewidth=2, label='Наша Вселенная')
        
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Индекс пригодности')
        ax1.set_title(f'Пригодность для жизни vs {param_name}')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # График 2: дискретные категории
        colors = ['darkred', 'red', 'orange', 'yellowgreen', 'darkgreen']
        labels = ['Мертвая', 'Враждебная', 'Маргинальная', 'Пригодная', 'Оптимальная']
        
        for i in range(5):
            mask = indices == i
            if np.any(mask):
                ax2.scatter(values[mask], [i]*np.sum(mask), 
                          c=colors[i], s=50, alpha=0.6, label=labels[i])
        
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Категория')
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(labels)
        ax2.set_title(f'Категории вселенных vs {param_name}')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.legend(loc='best')
        
        if our_val >= min(values) and our_val <= max(values):
            ax2.axvline(x=our_val, color='red', linewidth=2)
        
        plt.tight_layout()
        plt.show()
    
    def plot_2d_heatmap(self, result_2d: Dict, figsize: Tuple[int, int] = (10, 8)):
        """Строит тепловую карту 2D сканирования"""
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        im = ax.imshow(result_2d['score_map'].T, origin='lower', 
                      extent=[result_2d['values1'][0], result_2d['values1'][-1],
                             result_2d['values2'][0], result_2d['values2'][-1]],
                      aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        plt.colorbar(im, ax=ax, label='Индекс пригодности')
        
        ax.set_xlabel(result_2d['param1'])
        ax.set_ylabel(result_2d['param2'])
        ax.set_title(f'Пригодность для жизни: {result_2d["param1"]} vs {result_2d["param2"]}')
        
        # Отмечаем нашу Вселенную
        our_x = 1/137.036 if result_2d['param1'] == 'alpha' else 1.0
        our_y = 1/137.036 if result_2d['param2'] == 'alpha' else 1.0
        
        if (our_x >= result_2d['values1'][0] and our_x <= result_2d['values1'][-1] and
            our_y >= result_2d['values2'][0] and our_y <= result_2d['values2'][-1]):
            ax.plot(our_x, our_y, 'r*', markersize=15, label='Наша Вселенная')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, param_name: str, figsize: Tuple[int, int] = (10, 8)):
        """Строит матрицу корреляций между свойствами"""
        
        correlations = self.explorer.analyze_correlations(param_name)
        
        if not correlations:
            print("Нет данных для корреляционного анализа")
            return
        
        # Создаем матрицу корреляций между свойствами
        props = list(correlations.keys())
        n = len(props)
        corr_matrix = np.zeros((n, n))
        
        for i, p1 in enumerate(props):
            for j, p2 in enumerate(props):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Вычисляем корреляцию между свойствами
                    result = self.explorer.results[param_name]
                    values1 = [p[p1] for p in result['properties']]
                    values2 = [p[p2] for p in result['properties']]
                    corr_matrix[i, j] = np.corrcoef(values1, values2)[0, 1]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        plt.colorbar(im, ax=ax, label='Корреляция')
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(props, rotation=45, ha='right')
        ax.set_yticklabels(props)
        ax.set_title(f'Корреляции свойств при изменении {param_name}')
        
        # Добавляем значения в ячейки
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black' if abs(corr_matrix[i, j]) < 0.7 else 'white')
        
        plt.tight_layout()
        plt.show()

# ==================== ДЕМОНСТРАЦИЯ ====================

if __name__ == "__main__":
    
    print("="*60)
    print("🚀 МУЛЬТИВСЕЛЕННЫЙ ДИНАМИЧЕСКИЙ АНАЛИЗАТОР v2.0")
    print("="*60)
    
    # Создаем исследователя
    explorer = MultiverseDynamicsExplorer()
    visualizer = MultiverseVisualizer(explorer)
    
    # 1. Сканируем alpha в широком диапазоне
    explorer.scan_parameter(
        param_name="alpha",
        start=1/500,  # очень слабый электромагнетизм
        stop=1/20,    # очень сильный электромагнетизм
        num_points=200,
        log_scale=False
    )
    
    # 2. Визуализируем результаты
    visualizer.plot_habitability_scan("alpha")
    visualizer.plot_1d_scan("alpha")
    
    # 3. Находим критические точки
    critical = explorer.find_critical_points("alpha", threshold=0.5)
    print(f"\n🔍 Критические значения α (где пригодность падает ниже 0.5):")
    for i, val in enumerate(critical):
        print(f"   {i+1}. α = {val:.6f}")
    
    # 4. Анализ корреляций
    correlations = explorer.analyze_correlations("alpha")
    print(f"\n📊 Корреляции с α:")
    for prop, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {prop}: {corr:+.3f}")
    
    visualizer.plot_correlation_matrix("alpha")
    
    # 5. 2D сканирование (alpha vs масса протона)
    result_2d = explorer.scan_2d(
        param1_name="alpha",
        param1_range=(1/300, 1/30),
        param2_name="m_p",
        param2_range=(0.5, 2.0),  # масса протона относительно нашей
        num_points1=50,
        num_points2=50
    )
    
    visualizer.plot_2d_heatmap(result_2d)
    
    # 6. Детальный анализ нашей Вселенной
    print("\n" + "="*60)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ НАШЕЙ ВСЕЛЕННОЙ")
    print("="*60)
    
    our_analyzer = UniverseAnalyzer(UniverseParameters("🌍 Наша Вселенная"))
    index, score, metrics = our_analyzer.calculate_habitability_index()
    
    print(f"\n📊 Индекс пригодности: {score:.3f}")
    print(f"🏷️ Категория: {index.name}")
    print(f"\n📈 Метрики:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.2f}")
    
    # 7. Сравниваем несколько интересных вселенных
    print("\n" + "="*60)
    print("🌌 СРАВНЕНИЕ ИНТЕРЕСНЫХ ВСЕЛЕННЫХ")
    print("="*60)
    
    interesting_alphas = [1/300, 1/200, 1/137.036, 1/100, 1/50, 1/30]
    
    for alpha in interesting_alphas:
        u = UniverseParameters(name=f"α={alpha:.4f}", alpha=alpha)
        analyzer = UniverseAnalyzer(u)
        index, score, _ = analyzer.calculate_habitability_index()
        
        marker = "✅" if index.value >= HabitabilityIndex.HABITABLE.value else "⚠️" if index.value >= HabitabilityIndex.MARGINAL.value else "❌"
        print(f"{marker} {u.name}: {index.name} (score: {score:.3f})")
    
    # 8. Дополнительное сканирование для e и m_p
    print("\n" + "="*60)
    print("⚡ СКАНИРОВАНИЕ ЭЛЕМЕНТАРНОГО ЗАРЯДА")
    print("="*60)
    
    explorer.scan_parameter(
        param_name="e",
        start=0.1,    # 10% от нашего заряда
        stop=3.0,     # 300% от нашего заряда
        num_points=100,
        log_scale=False
    )
    
    visualizer.plot_habitability_scan("e")
    
    print("\n" + "="*60)
    print("🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("="*60)
