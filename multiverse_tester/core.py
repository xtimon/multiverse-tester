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
    H_0: float = 2.2e-18  # с⁻¹ (70 км/с/Мпк)
    Lambda: float = 1.1e-52  # м⁻² (космологическая постоянная)
    # Атомные массы (в кг)
    m_he4: float = 6.6464764e-27  # кг (4He)
    m_c12: float = 1.9926467e-26  # кг (12C)
    m_o16: float = 2.6560178e-26  # кг (16O)
    m_ne20: float = 3.3208645e-26  # кг (20Ne)
    m_si28: float = 4.6467789e-26  # кг (28Si)
    m_fe56: float = 9.2882735e-26  # кг (56Fe)

class UniverseParameters:
    """Полное описание Вселенной"""
    
    def __init__(self, name="Our Universe", alpha=None, e=None, m_p=None,
                 m_e=None, hbar=None, c=None, G=None, epsilon_0=None, k_B=None,
                 H_0=None, Lambda=None):
        self.name = name
        self.const = UniversalConstants()
        
        self.k_B = k_B if k_B is not None else self.const.k_B
        self.const.k_B = self.k_B
        self.H_0 = H_0 if H_0 is not None else self.const.H_0
        self.const.H_0 = self.H_0
        self.Lambda = Lambda if Lambda is not None else self.const.Lambda
        self.const.Lambda = self.Lambda
        self.hbar = hbar if hbar else self.const.hbar
        self.c = c if c else self.const.c
        self.G = G if G else self.const.G
        self.epsilon_0 = epsilon_0 if epsilon_0 else self.const.epsilon_0
        self.m_e = m_e if m_e else self.const.m_e
        
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
        
        # Масштабируем массы ядер пропорционально m_p
        self.m_he4 = self.const.m_he4 * (self.m_p / self.const.m_p)
        self.m_c12 = self.const.m_c12 * (self.m_p / self.const.m_p)
        self.m_o16 = self.const.m_o16 * (self.m_p / self.const.m_p)
        self.m_ne20 = self.const.m_ne20 * (self.m_p / self.const.m_p)
        self.m_si28 = self.const.m_si28 * (self.m_p / self.const.m_p)
        self.m_fe56 = self.const.m_fe56 * (self.m_p / self.const.m_p)
    
    def __repr__(self):
        return f"{self.name}: α={self.alpha:.6f}, e/e₀={self.e/self.const.e:.3f}, m_p/m_p₀={self.m_p/self.const.m_p:.3f}"

# ==================== АТОМНАЯ ФИЗИКА ====================

class AtomicPhysics:
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def bohr_radius(self) -> float:
        return (4 * math.pi * self.u.epsilon_0 * self.u.hbar**2) / (self.u.m_e * self.u.e**2)
    
    def rydberg_energy(self) -> float:
        return (self.u.m_e * self.u.e**4) / (32 * math.pi**2 * self.u.epsilon_0**2 * self.u.hbar**2)
    
    def rydberg_ev(self) -> float:
        return self.rydberg_energy() / self.u.const.e
    
    def compton_wavelength(self) -> float:
        return self.u.hbar / (self.u.m_e * self.u.c)
    
    def fine_structure_effects(self) -> Dict[str, float]:
        a0 = self.bohr_radius()
        λ_c = self.compton_wavelength()
        return {
            'a0': a0,
            'a0_norm': a0 / 5.29e-11,
            'a0_over_λc': a0 / λ_c,
            'E_bind': self.rydberg_ev(),
            'E_bind_norm': self.rydberg_ev() / 13.6
        }

# ==================== ЯДЕРНАЯ ФИЗИКА ====================

class NuclearPhysics:
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def qcd_scale(self, alpha_dependence: float = 0.1) -> float:
        base_lambda = 2.5e-28
        alpha_ratio = self.u.alpha / (1/137.036)
        correction = 1 + alpha_dependence * math.log(alpha_ratio) if alpha_ratio > 0 else 1
        return base_lambda * max(0.3, min(3.0, correction))
    
    def binding_energy(self, A: int, Z: int, alpha_dependence: float = 0.1) -> float:
        """Энергия связи ядра (полуэмпирическая формула)"""
        # Коэффициенты для нашей Вселенной (в МэВ)
        a_v = 15.75  # объемный член
        a_s = 17.8   # поверхностный член
        a_c = 0.711  # кулоновский член (пропорционален α)
        a_a = 23.7   # асимметрия
        a_p = 11.18  # спаривание
        
        # Масштабируем кулоновский член с α
        a_c_scaled = a_c * (self.u.alpha / (1/137.036))
        
        # Объемный член зависит от масштаба КХД
        qcd_scale_factor = self.qcd_scale(alpha_dependence) / 2.5e-28
        a_v_scaled = a_v * qcd_scale_factor
        a_s_scaled = a_s * qcd_scale_factor
        
        # Четность-нечетность
        if A % 2 == 0:
            if Z % 2 == 0:
                delta = a_p / A**(3/4)  # четно-четное
            else:
                delta = -a_p / A**(3/4)  # нечетно-нечетное
        else:
            delta = 0  # нечетное A
        
        # Формула Вайцзеккера
        B = (a_v_scaled * A - 
             a_s_scaled * A**(2/3) - 
             a_c_scaled * Z**2 / A**(1/3) - 
             a_a * (A - 2*Z)**2 / A +
             delta)
        
        return B * 1e6 * self.u.const.e  # в Дж
    
    def binding_per_nucleon(self, A: int, Z: int) -> float:
        """Энергия связи на нуклон в МэВ"""
        return self.binding_energy(A, Z) / (A * self.u.const.e * 1e6)
    
    def coulomb_barrier(self, Z1: int, Z2: int, A1: int, A2: int) -> float:
        """Кулоновский барьер для слияния ядер"""
        r0 = 1.2e-15  # ферми
        r = r0 * (A1**(1/3) + A2**(1/3))
        barrier = (Z1 * Z2 * self.u.alpha * self.u.hbar * self.u.c) / (4 * math.pi * r)
        return barrier  # в Дж
    
    def neutron_drip_line(self, Z: int) -> int:
        """Приблизительная граница нейтронной стабильности"""
        # В нашей Вселенной: N ~ 1.5 * Z для тяжелых ядер
        # Зависит от баланса кулоновского отталкивания и сильного взаимодействия
        coulomb_factor = (self.u.alpha / (1/137.036))
        return int(Z * (1.2 + 0.3 * coulomb_factor))

# ==================== ЗВЕЗДНЫЙ НУКЛЕОСИНТЕЗ (РАСШИРЕННАЯ ВЕРСИЯ) ====================

# Референсные скорости для нашей Вселенной (для rate_relative — сравнение с нашей)
_REF_RATES = None

def _get_ref_rates():
    """Вычисляет эталонные скорости один раз для нашей Вселенной (без рекурсии)"""
    global _REF_RATES
    if _REF_RATES is None:
        ref_u = UniverseParameters(name="Reference")
        kT_pp = ref_u.const.k_B * 1.5e7
        m_red_pp = ref_u.m_p / 2
        E_G_pp = (math.pi * ref_u.alpha)**2 * (m_red_pp * ref_u.c**2 / 2)
        gamow_pp = math.exp(-math.sqrt(E_G_pp / kT_pp))
        pp_rate = ref_u.alpha**2 * (kT_pp)**(2/3) * gamow_pp

        kT_cno = ref_u.const.k_B * 2e7
        m_red_cno = ref_u.m_p * 14 / 15
        E_G_cno = (math.pi * ref_u.alpha * 7)**2 * (m_red_cno * ref_u.c**2 / 2)
        gamow_cno = math.exp(-math.sqrt(E_G_cno / kT_cno))
        cno_rate = 0.01 * ref_u.alpha * gamow_cno

        kT_ta = ref_u.const.k_B * 1e8
        E_res = 7.65e6 * ref_u.const.e * (ref_u.alpha / (1/137.036))**2
        E_3alpha = 3 * kT_ta
        gamma_res = 10e3 * ref_u.const.e
        resonance_factor = (gamma_res/2)**2 / ((E_res - E_3alpha)**2 + (gamma_res/2)**2)
        Q_Be = 92e3 * ref_u.const.e
        K_eq = math.exp(-Q_Be / kT_ta) * (ref_u.alpha / (1/137.036))**3
        ta_rate = K_eq * resonance_factor * ref_u.alpha**3

        kT_c = ref_u.const.k_B * 8e8
        m_red_c = 6 * ref_u.m_p
        E_G_c = (math.pi * ref_u.alpha * 36)**2 * (m_red_c * ref_u.c**2 / 2)
        gamow_c = math.exp(-math.sqrt(E_G_c / kT_c))
        carbon_rate = ref_u.alpha**2 * gamow_c

        _REF_RATES = {'pp': pp_rate, 'cno': cno_rate, 'triple_alpha': ta_rate, 'carbon': carbon_rate}
    return _REF_RATES


class StellarNucleosynthesis:
    """Расширенный класс для всех процессов звездного нуклеосинтеза"""
    
    def __init__(self, universe: UniverseParameters, nuclear: NuclearPhysics):
        self.u = universe
        self.nuclear = nuclear
        
    # ===== Водородное горение =====
    
    def pp_chain_rate(self, T: float = 1.5e7) -> Dict[str, float]:
        """
        pp-цепочка (основная в звездах типа Солнца)
        
        Аргументы:
            T: температура в K
        """
        kT = self.u.const.k_B * T
        m_reduced = self.u.m_p / 2  # приведённая масса p+p
        
        # Гамов-фактор для туннелирования
        E_G = (math.pi * self.u.alpha)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        # Скорость реакции
        rate = self.u.alpha**2 * (kT)**(2/3) * gamow
        ref = _get_ref_rates()['pp']
            
        # Время жизни водорода в звезде (упрощенно)
        tau_h = 1e10 / rate * ref  # в годах
            
        return {
            'rate_relative': rate / ref,
            'gamow_factor': gamow,
            'tau_hydrogen_years': tau_h,
            'barrier_mev': self.nuclear.coulomb_barrier(1, 1, 1, 1) / (self.u.const.e * 1e6)
        }
    
    def pep_reaction(self, T: float = 1.5e7) -> float:
        """Реакция p + e⁻ + p → d + ν (pep-цепочка)"""
        # Зависит от плотности электронов и α
        electron_density_factor = self.u.alpha**3  # упрощенно
        return self.pp_chain_rate(T)['rate_relative'] * electron_density_factor
    
    def cno_cycle_rate(self, T: float = 2e7) -> Dict[str, float]:
        """
        CNO-цикл (доминирует в массивных звездах)
        
        Зависит от обилия C,N,O и кулоновских барьеров
        """
        kT = self.u.const.k_B * T
        
        # Средний заряд для C,N,O с протоном
        Z_avg = 7
        A_avg = 14
        
        # Энергия Гамова (приведённая масса)
        m_reduced = self.u.m_p * A_avg / (A_avg + 1)
        E_G = (math.pi * self.u.alpha * Z_avg)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        # Скорость реакции (пропорциональна обилию CNO и α)
        abundance_factor = 0.01  # типичное обилие CNO в звездах
        rate = abundance_factor * self.u.alpha * gamow
        ref = _get_ref_rates()['cno']
        
        # Температурная чувствительность
        dlnr_dlnT = (E_G / kT)**0.5 / 3
        
        return {
            'rate_relative': rate / ref,
            'gamow_factor': gamow,
            'T_sensitivity': dlnr_dlnT
        }
    
    # ===== Гелиевое горение =====
    
    def triple_alpha(self, T: float = 1e8) -> Dict[str, float]:
        """
        Тройная альфа-реакция (образование углерода)
        
        4He + 4He ↔ 8Be
        8Be + 4He → 12C + γ
        
        Критически зависит от резонанса Хойла в 12C
        """
        kT = self.u.const.k_B * T
        
        # Энергия резонанса Хойла в нашей Вселенной (7.65 МэВ выше основного состояния)
        E_res_our = 7.65e6 * self.u.const.e  # в Дж
        
        # В нашей Вселенной резонанс близок к энергии 3α при T~1e8 K
        # Сдвиг резонанса с α (приблизительно квадратичная зависимость)
        E_res = E_res_our * (self.u.alpha / (1/137.036))**2
        
        # Энергия трех α-частиц при температуре T
        E_3alpha = 3 * kT  # приблизительно
        
        # Насколько близко резонанс к E_3alpha
        # Используем распределение Брейта-Вигнера
        gamma_res = 10e3 * self.u.const.e  # ширина резонанса ~10 кэВ
        resonance_factor = (gamma_res/2)**2 / ((E_res - E_3alpha)**2 + (gamma_res/2)**2)
        
        # Концентрация 8Be (равновесная)
        # Q-значение реакции 4He+4He↔8Be (~92 кэВ в нашей Вселенной)
        Q_Be = 92e3 * self.u.const.e
        # Константа равновесия зависит от α (через кулон)
        K_eq = math.exp(-Q_Be / kT) * (self.u.alpha / (1/137.036))**3
        
        # Скорость тройной альфа
        rate = K_eq * resonance_factor * self.u.alpha**3
        ref = _get_ref_rates()['triple_alpha']
        
        return {
            'rate_relative': rate / ref,
            'resonance_energy_kev': E_res / self.u.const.e / 1000,
            'resonance_factor': resonance_factor,
            'Be_abundance': K_eq,
            'carbon_production': "Высокое" if resonance_factor > 0.1 else "Низкое"
        }
    
    def alpha_capture(self, target_Z: int, target_A: int, T: float = 2e8) -> float:
        """
        Реакции захвата α-частиц:
        12C(α,γ)16O
        16O(α,γ)20Ne
        20Ne(α,γ)24Mg и т.д.
        """
        kT = self.u.const.k_B * T
        Z_target = target_Z
        A_target = target_A
        
        # Приведенная масса (α + ядро-мишень)
        m_reduced = (4 * A_target) / (4 + A_target) * self.u.m_p
        
        # Кулоновский барьер
        E_G = (math.pi * self.u.alpha * 2 * Z_target)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        # Скорость захвата
        rate = self.u.alpha * gamow
        
        return rate
    
    # ===== Горение углерода и кислорода =====
    
    def carbon_burning(self, T: float = 8e8) -> Dict[str, float]:
        """
        Горение углерода: 12C + 12C → 20Ne + α, 23Na + p, 23Mg + n
        Происходит в массивных звездах после исчерпания гелия
        """
        kT = self.u.const.k_B * T
        
        # Кулоновский барьер для C+C
        Z = 6
        A = 12
        m_reduced = A/2 * self.u.m_p
        E_G = (math.pi * self.u.alpha * Z**2)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        # Скорость реакции (сильно зависит от α и m_p через Gamow)
        rate = self.u.alpha**2 * gamow
        ref = _get_ref_rates()['carbon']
        
        # Температура зажигания
        T_ignition = 8e8 * (self.u.alpha / (1/137.036))**2  # приблизительно
        
        return {
            'rate_relative': rate / ref,
            'T_ignition': T_ignition,
            'gamow': gamow
        }
    
    def oxygen_burning(self, T: float = 2e9) -> float:
        """
        Горение кислорода: 16O + 16O → 28Si + α, 31P + p, 31S + n
        """
        kT = self.u.const.k_B * T
        
        Z = 8
        A = 16
        m_reduced = A/2 * self.u.m_p
        E_G = (math.pi * self.u.alpha * Z**2)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        return self.u.alpha**2 * gamow
    
    # ===== Кремниевое горение и альфа-процесс =====
    
    def silicon_burning(self, T: float = 3e9) -> float:
        """
        Кремниевое горение: фотодиссоциация и альфа-захват
        В результате образуется железо
        """
        kT = self.u.const.k_B * T
        
        # Равновесие между фотодиссоциацией и захватом
        # Q-значение для реакции 28Si + γ ↔ 24Mg + α
        Q_si = 10e6 * self.u.const.e  # ~10 МэВ
        
        # Константа равновесия зависит от α
        K_eq = math.exp(-Q_si / kT) * (self.u.alpha / (1/137.036))**3
        
        return K_eq
    
    def alpha_process(self, T: float = 2e9) -> List[Dict]:
        """
        Альфа-процесс: последовательный захват α-частиц
        от Ne до Fe
        
        Возвращает список ядер и вероятности их образования
        """
        nuclei = [
            {'name': '20Ne', 'Z': 10, 'A': 20},
            {'name': '24Mg', 'Z': 12, 'A': 24},
            {'name': '28Si', 'Z': 14, 'A': 28},
            {'name': '32S', 'Z': 16, 'A': 32},
            {'name': '36Ar', 'Z': 18, 'A': 36},
            {'name': '40Ca', 'Z': 20, 'A': 40},
            {'name': '44Ti', 'Z': 22, 'A': 44},
            {'name': '48Cr', 'Z': 24, 'A': 48},
            {'name': '52Fe', 'Z': 26, 'A': 52},
            {'name': '56Ni', 'Z': 28, 'A': 56}
        ]
        
        results = []
        prev_rate = 1.0
        
        for i, nucleus in enumerate(nuclei):
            if i == 0:
                # Начинаем с Ne
                rate = self.alpha_capture(10, 20, T)
            else:
                # Последовательный захват
                rate = prev_rate * self.alpha_capture(nucleus['Z'], nucleus['A'], T)
            
            results.append({
                'nucleus': nucleus['name'],
                'relative_yield': rate,
                'Z': nucleus['Z'],
                'A': nucleus['A']
            })
            prev_rate = rate
        
        return results
    
    # ===== Нейтронные процессы =====
    
    def s_process(self, T: float = 3e8, neutron_density: float = 1e8) -> Dict[str, float]:
        """
        s-процесс (медленный захват нейтронов)
        Происходит в AGB-звездах
        
        Зависит от:
        - сечения захвата нейтронов (слабо зависит от α)
        - времени между захватами (зависит от β-распадов)
        """
        # Сечение захвата нейтронов ~ 1/v ~ 1/sqrt(T)
        cross_section = 1.0 / math.sqrt(T / 3e8)
        
        # β-распады зависят от энергии Ферми и α
        # Время жизни относительно β-распада
        beta_lifetime = 1e3 * (self.u.alpha / (1/137.036))**2  # в годах, упрощенно
        
        # Время между захватами нейтронов
        capture_time = 1e5 / (neutron_density * cross_section)  # годы
        
        # Путь s-процесса (какие ядра успевают образоваться)
        if capture_time < beta_lifetime:
            path = "медленный (s-процесс)"
            abundance = "высокое"
        else:
            path = "быстрый (r-процесс)"
            abundance = "низкое для s-процесса"
        
        return {
            'path': path,
            'capture_time_years': capture_time,
            'beta_lifetime_years': beta_lifetime,
            'abundance': abundance
        }
    
    def r_process(self, neutron_density: float = 1e20, T: float = 1e9) -> Dict[str, float]:
        """
        r-процесс (быстрый захват нейтронов)
        Происходит в сверхновых и нейтронных звездах
        
        Очень высокая плотность нейтронов
        """
        # При очень высокой плотности нейтронов
        # захват идет до нейтронно-избыточных ядер
        
        # Положение линии капель нейтронов
        neutron_drip = []
        heavy_elements = []
        
        for Z in range(26, 92):  # от Fe до U
            N_max = self.nuclear.neutron_drip_line(Z)
            neutron_drip.append(N_max)
            
            # Оцениваем обилие трансурановых элементов
            if Z > 92:
                heavy_elements.append(Z)
        
        return {
            'max_neutron_excess': max(neutron_drip) if neutron_drip else 0,
            'transuranic_elements': len(heavy_elements),
            'r_process_abundance': "высокое" if neutron_density > 1e18 else "низкое"
        }
    
    # ===== Взрывной нуклеосинтез =====
    
    def supernova_nucleosynthesis(self, progenitor_mass: float = 15) -> Dict[str, List]:
        """
        Нуклеосинтез в сверхновой (очень упрощенно)
        
        Аргументы:
            progenitor_mass: масса предшественника в массах Солнца
        """
        # Элементы, образующиеся в разных слоях
        elements = {
            'outer_H': ['H', 'He'],
            'He_layer': ['He', 'C', 'O'],
            'C_layer': ['C', 'O', 'Ne', 'Mg'],
            'O_layer': ['O', 'Mg', 'Si', 'S'],
            'Si_layer': ['Si', 'S', 'Ar', 'Ca'],
            'Fe_core': ['Fe', 'Co', 'Ni']
        }
        
        # Зависимость от α и G: масса железного ядра
        # Больше α → сильнее кулон → меньше железное ядро
        # M_Ch ~ G^(-1.5): меньше G → больше предельная масса белого карлика
        G_ratio = self.u.G / self.u.const.G
        M_fe_core = 1.4 * (self.u.alpha / (1/137.036))**(-0.5) * (G_ratio)**(-0.5)  # массы Солнца
        
        # Порог коллапса
        if M_fe_core > 1.8:  # слишком большое ядро
            collapse_type = "черная дыра"
            neutron_star = False
        elif M_fe_core < 1.0:  # слишком маленькое ядро
            collapse_type = "белый карлик"
            neutron_star = False
        else:
            collapse_type = "нейтронная звезда"
            neutron_star = True
        
        return {
            'elements': elements,
            'fe_core_mass': M_fe_core,
            'collapse_type': collapse_type,
            'neutron_star_formed': neutron_star,
            'r_process_possible': neutron_star  # в нейтронных звездах возможен r-процесс
        }
    
    def complete_nucleosynthesis_analysis(self) -> Dict:
        """Полный анализ всех процессов нуклеосинтеза"""
        
        results = {}
        
        # Водородное горение
        results['pp_chain'] = self.pp_chain_rate()
        results['cno_cycle'] = self.cno_cycle_rate()
        
        # Гелиевое горение
        results['triple_alpha'] = self.triple_alpha()
        
        # Углеродное горение
        results['carbon_burning'] = self.carbon_burning()
        
        # Альфа-процесс
        results['alpha_process'] = self.alpha_process()
        
        # Нейтронные процессы
        results['s_process'] = self.s_process()
        results['r_process'] = self.r_process()
        
        # Сверхновые
        results['supernova'] = self.supernova_nucleosynthesis()
        
        # Оцениваем общее производство элементов
        element_production = {
            'H_He': results['pp_chain']['rate_relative'] * results['cno_cycle']['rate_relative'],
            'C': results['triple_alpha']['rate_relative'],
            'O_Mg_Si': np.mean([r['relative_yield'] for r in results['alpha_process'][:5]]),
            'Fe_peak': results['carbon_burning']['rate_relative'] * results['supernova']['fe_core_mass'],
            'heavy_r_process': len(results['r_process']) if results['r_process']['transuranic_elements'] > 0 else 0
        }
        
        results['element_production'] = element_production
        
        return results

# ==================== ИНДЕКС ПРИГОДНОСТИ ====================

class HabitabilityIndex(Enum):
    DEAD = 0
    HOSTILE = 1
    MARGINAL = 2
    HABITABLE = 3
    OPTIMAL = 4

class UniverseAnalyzer:
    """Полный анализ Вселенной с расширенным нуклеосинтезом"""
    
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        self.atomic = AtomicPhysics(universe)
        self.nuclear = NuclearPhysics(universe)
        self.stellar = StellarNucleosynthesis(universe, self.nuclear)
        
    def calculate_habitability_index(self) -> Tuple[HabitabilityIndex, float, Dict]:
        """Вычисляет индекс пригодности для жизни"""
        
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
        if 1/200 < α < 1/50:
            metrics['chemistry'] = 1.0
        elif 1/300 < α < 1/30:
            metrics['chemistry'] = 0.5
        else:
            metrics['chemistry'] = 0.0
        
        # 3. Ядерная энергия
        binding_fe = self.nuclear.binding_per_nucleon(56, 26)
        if 7 < binding_fe < 10:
            metrics['nuclear'] = 1.0
        elif 4 < binding_fe < 12:
            metrics['nuclear'] = 0.5
        else:
            metrics['nuclear'] = 0.0
        
        # 4. Звездный синтез (полный анализ)
        nucleosynthesis = self.stellar.complete_nucleosynthesis_analysis()
        
        # Углерод
        carbon_prod = nucleosynthesis['triple_alpha']['rate_relative']
        if carbon_prod > 0.5:
            metrics['carbon'] = 1.0
        elif carbon_prod > 0.1:
            metrics['carbon'] = 0.5
        else:
            metrics['carbon'] = 0.0
        
        # Тяжелые элементы (альфа-процесс)
        alpha_prod = np.mean([r['relative_yield'] for r in nucleosynthesis['alpha_process'][:5]])
        if alpha_prod > 0.3:
            metrics['heavy_elements'] = 1.0
        elif alpha_prod > 0.1:
            metrics['heavy_elements'] = 0.5
        else:
            metrics['heavy_elements'] = 0.0
        
        # Водородный синтез
        pp_rate = nucleosynthesis['pp_chain']['rate_relative']
        if 0.1 < pp_rate < 10:
            metrics['fusion'] = 1.0
        elif 0.01 < pp_rate < 100:
            metrics['fusion'] = 0.5
        else:
            metrics['fusion'] = 0.0
        
        # Сверхновые (образование нейтронных звезд)
        if nucleosynthesis['supernova']['neutron_star_formed']:
            metrics['supernova'] = 1.0
        else:
            metrics['supernova'] = 0.0
        
        # r-процесс (тяжелые элементы)
        if nucleosynthesis['r_process']['transuranic_elements'] > 0:
            metrics['r_process'] = 1.0
        else:
            metrics['r_process'] = 0.0
        
        # 9. Космология (H₀, Λ)
        t_Hubble = 1.0 / self.u.H_0 if self.u.H_0 > 0 else 1e20  # с
        t_Gyr = t_Hubble / 3.15e16
        if 2 < t_Gyr < 100:  # 2–100 млрд лет
            metrics['cosmology_H'] = 1.0
        elif 0.5 < t_Gyr < 200:
            metrics['cosmology_H'] = 0.5
        else:
            metrics['cosmology_H'] = 0.0
        Lambda_ratio = self.u.Lambda / self.u.const.Lambda if self.u.const.Lambda != 0 else 1.0
        if 0.1 < Lambda_ratio < 50:
            metrics['cosmology_Lambda'] = 1.0
        elif 0.01 < Lambda_ratio < 200:
            metrics['cosmology_Lambda'] = 0.5
        else:
            metrics['cosmology_Lambda'] = 0.0
        
        # Вычисляем общий индекс
        weights = {
            'atomic': 0.14,
            'chemistry': 0.19,
            'nuclear': 0.14,
            'carbon': 0.14,
            'heavy_elements': 0.09,
            'fusion': 0.09,
            'supernova': 0.05,
            'r_process': 0.09,
            'cosmology_H': 0.04,
            'cosmology_Lambda': 0.04
        }
        
        total_score = sum(metrics.get(k, 0) * weights[k] for k in weights)
        
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
        nucleo = self.stellar.complete_nucleosynthesis_analysis()
        
        return {
            'alpha': self.u.alpha,
            'e_ratio': self.u.e / self.u.const.e,
            'bohr_radius_norm': atomic['a0_norm'],
            'binding_energy_mev': self.nuclear.binding_per_nucleon(56, 26),
            'pp_rate': nucleo['pp_chain']['rate_relative'],
            'triple_alpha_rate': nucleo['triple_alpha']['rate_relative'],
            'carbon_prod': nucleo['triple_alpha']['rate_relative'],
            'alpha_process_yield': np.mean([r['relative_yield'] for r in nucleo['alpha_process'][:5]])
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
        """Сканирует один параметр"""
        
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
                param_values.append(val)
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
            
            analyzer = UniverseAnalyzer(u)
            props = analyzer.get_all_properties()
            index, score, metrics = analyzer.calculate_habitability_index()
            
            properties_list.append(props)
            indices.append(index.value)
            scores.append(score)
            
            if i % max(1, num_points//10) == 0:
                print(f"   Прогресс: {i}/{num_points} ({i/num_points*100:.1f}%)")
        
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
    
    def analyze_correlations(self, param_name: str) -> Dict:
        """Анализирует корреляции"""
        if param_name not in self.results:
            raise ValueError(f"Сначала выполните сканирование для {param_name}")
        
        result = self.results[param_name]
        values = result['param_values']
        props = result['properties']
        
        if not props:
            return {}
        
        keys = props[0].keys()
        correlations = {}
        
        for key in keys:
            prop_values = [p[key] for p in props]
            if not np.all(np.isnan(prop_values)):
                corr = np.corrcoef(values, prop_values)[0, 1]
                correlations[key] = corr if not np.isnan(corr) else 0.0
        
        return correlations

