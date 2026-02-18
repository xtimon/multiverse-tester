import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Optional

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
    e: float = 1.60217662e-19  # Кл (элементарный заряд) ← ДОБАВЛЕНО!

class UniverseParameters:
    """Полное описание Вселенной со всеми взаимосвязями"""
    
    def __init__(self, name="Our Universe", alpha=None, e=None, m_p=None, 
                 hbar=None, c=None, G=None, epsilon_0=None):
        self.name = name
        self.const = UniversalConstants()
        
        # Базовые константы (с приоритетом пользовательских значений)
        self.hbar = hbar if hbar else self.const.hbar
        self.c = c if c else self.const.c
        self.G = G if G else self.const.G
        self.epsilon_0 = epsilon_0 if epsilon_0 else self.const.epsilon_0
        
        # Установка alpha и e (взаимосвязь)
        if alpha is not None:
            self.alpha = alpha
            self.e = math.sqrt(alpha * 4 * math.pi * self.epsilon_0 * self.hbar * self.c)
        elif e is not None:
            self.e = e
            self.alpha = (e**2) / (4 * math.pi * self.epsilon_0 * self.hbar * self.c)
        else:
            # Наша Вселенная по умолчанию
            self.e = self.const.e
            self.alpha = (self.e**2) / (4 * math.pi * self.epsilon_0 * self.hbar * self.c)
        
        # Масса протона
        if m_p is not None:
            self.m_p = m_p
        else:
            self.m_p = self.const.m_p
        
        # Планковские единицы
        self.m_planck = math.sqrt(self.hbar * self.c / self.G)
        self.l_planck = math.sqrt(self.hbar * self.G / self.c**3)
        self.t_planck = self.l_planck / self.c
        self.q_planck = math.sqrt(4 * math.pi * self.epsilon_0 * self.hbar * self.c)
    
    def __repr__(self):
        # Исправлено: используем self.e и self.const.e
        return f"{self.name}: α = {self.alpha:.6f}, e/e₀ = {self.e/self.const.e:.3f}, m_p/m_p₀ = {self.m_p/self.const.m_p:.3f}"

class AtomicPhysics:
    """Атомная физика: структура атомов и молекул"""
    
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def bohr_radius(self) -> float:
        """Радиус Бора (размер атома водорода)"""
        return (4 * math.pi * self.u.epsilon_0 * self.u.hbar**2) / (self.u.const.m_e * self.u.e**2)
    
    def rydberg_energy(self) -> float:
        """Энергия связи электрона в атоме водорода (Дж)"""
        return (self.u.const.m_e * self.u.e**4) / (32 * math.pi**2 * self.u.epsilon_0**2 * self.u.hbar**2)
    
    def rydberg_ev(self) -> float:
        """Энергия связи в электрон-вольтах"""
        return self.rydberg_energy() / self.u.const.e  # Исправлено: используем self.u.const.e
    
    def compton_wavelength(self) -> float:
        """Комптоновская длина волны электрона"""
        return self.u.hbar / (self.u.const.m_e * self.u.c)
    
    def atomic_timescale(self) -> float:
        """Характерное время атомных процессов"""
        return self.u.hbar / self.rydberg_energy()
    
    def critical_alpha_for_relativity(self) -> float:
        """Alpha, при которой релятивистские эффекты становятся критическими"""
        return 1.0  # Упрощенно

class NuclearPhysics:
    """Ядерная физика: сильное взаимодействие и структура ядер"""
    
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def qcd_scale(self) -> float:
        """Масштаб КХД (Lambda_QCD)"""
        base_lambda = 2.5e-28  # кг (~250 МэВ)
        alpha_ratio = self.u.alpha / (1/137.036)
        correction = 1 + 0.1 * math.log(alpha_ratio) if alpha_ratio > 0 else 1
        return base_lambda * max(0.5, min(2.0, correction))
    
    def proton_mass_qcd(self) -> float:
        """Масса протона, определяемая КХД"""
        return self.qcd_scale() * 6.5
    
    def binding_energy_per_nucleon(self, A: int = 56) -> float:
        """Энергия связи на нуклон (для железа)"""
        e0 = 8.5e6 * self.u.const.e  # 8.5 МэВ в Дж, исправлено: используем self.u.const.e
        
        Z = A/2 if A <= 56 else 26
        coulomb_term = (self.u.alpha / (1/137.036)) * (Z**2) / (A**(4/3))
        strong_term = self.qcd_scale() / 2.5e-28
        
        binding = e0 * (strong_term - 0.1 * coulomb_term)
        return max(0, binding)
    
    def coulomb_barrier(self, Z1: int, Z2: int) -> float:
        """Кулоновский барьер для слияния ядер"""
        r_nucleus = 1.2e-15 * ((Z1 + Z2) ** (1/3))
        barrier = (Z1 * Z2 * self.u.alpha * self.u.hbar * self.u.c) / (4 * math.pi * r_nucleus)
        return barrier

class StellarNucleosynthesis:
    """Звездный нуклеосинтез: образование элементов в звездах"""
    
    def __init__(self, universe: UniverseParameters, atomic: AtomicPhysics, nuclear: NuclearPhysics):
        self.u = universe
        self.atomic = atomic
        self.nuclear = nuclear
        self._our_rate = None  # Для нормализации
        
    def triple_alpha_resonance(self) -> Tuple[float, str]:
        """Анализ тройной гелиевой реакции"""
        energy_above_ground = 380e3 * self.u.const.e  # 380 кэВ в Дж, исправлено
        
        resonance_shift = (self.u.alpha / (1/137.036))**2
        actual_energy = energy_above_ground * resonance_shift
        
        kT = self.u.const.k_B * 1e8
        
        if abs(actual_energy - energy_above_ground) < kT:
            resonance_quality = "Отличный резонанс, производство углерода: Высокая"
        elif abs(actual_energy - energy_above_ground) < 10 * kT:
            resonance_quality = "Умеренный резонанс, производство углерода: Средняя"
        else:
            resonance_quality = "Резонанс отсутствует, производство углерода: Очень низкая"
        
        return actual_energy / self.u.const.e / 1000, resonance_quality  # в кэВ
    
    def proton_proton_chain(self) -> Dict[str, float]:
        """pp-цепочка: основной источник энергии в звездах типа Солнца"""
        T_sun = 1.5e7  # K
        kT = self.u.const.k_B * T_sun
        
        m_reduced = self.u.const.m_p / 2
        E_G = (math.pi * self.u.alpha)**2 * (m_reduced * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        rate = self.u.alpha**2 * (kT)**(2/3) * gamow
        
        if self._our_rate is None:
            self._our_rate = rate
        
        return {
            'rate_relative': rate / self._our_rate,
            'gamow_factor': gamow,
            'barrier_mev': self.nuclear.coulomb_barrier(1, 1) / (self.u.const.e * 1e6)
        }
    
    def cno_cycle(self) -> float:
        """CNO-цикл для массивных звезд"""
        avg_product = 7
        T_massive = 3e7  # K
        kT = self.u.const.k_B * T_massive
        
        E_G = (math.pi * self.u.alpha * avg_product)**2 * (self.u.const.m_p * self.u.c**2 / 2)
        gamow = math.exp(-math.sqrt(E_G / kT))
        
        return self.u.alpha * gamow

class GravitationalPhysics:
    """Гравитационные эффекты и связь с массами"""
    
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        
    def gravitational_coupling(self) -> float:
        """Гравитационная константа связи для протона"""
        return self.u.G * self.u.m_p**2 / (self.u.hbar * self.u.c)
    
    def proton_to_planck_mass_ratio(self) -> float:
        """Отношение массы протона к планковской массе"""
        return self.u.m_p / self.u.m_planck
    
    def schwarzschild_radius(self, mass: float = None) -> float:
        """Шварцшильдовский радиус для заданной массы"""
        if mass is None:
            mass = self.u.m_p
        return 2 * self.u.G * mass / self.u.c**2
    
    def planck_star_condition(self, mass: float) -> bool:
        """Достигает ли объект планковской плотности?"""
        r_s = self.schwarzschild_radius(mass)
        density = mass / (4/3 * math.pi * r_s**3)
        planck_density = self.u.m_planck / self.u.l_planck**3
        return density > planck_density

class UniverseStabilityAnalyzer:
    """Полный анализ стабильности и пригодности Вселенной для жизни"""
    
    def __init__(self, universe: UniverseParameters):
        self.u = universe
        self.atomic = AtomicPhysics(universe)
        self.nuclear = NuclearPhysics(universe)
        self.stellar = StellarNucleosynthesis(universe, self.atomic, self.nuclear)
        self.grav = GravitationalPhysics(universe)
        
    def analyze_all(self) -> Dict[str, Tuple[bool, str]]:
        """Запускает все проверки и возвращает результаты"""
        results = {}
        
        # 1. Атомная стабильность
        a0 = self.atomic.bohr_radius()
        λ_c = self.atomic.compton_wavelength()
        if a0 < λ_c:
            results['atomic'] = (False, f"Атомы коллапсируют: a0/λ_c = {a0/λ_c:.2f} < 1")
        elif a0 > 1000 * λ_c:
            results['atomic'] = (False, f"Атомы слишком диффузны: a0/λ_c = {a0/λ_c:.2e}")
        else:
            results['atomic'] = (True, f"Атомы стабильны: a0/λ_c = {a0/λ_c:.2f}")
        
        # 2. Химическая сложность
        α = self.u.alpha
        if α < 1/300:
            results['chemistry'] = (False, f"Химические связи слишком слабы: α={α:.4f} < 0.0033")
        elif α > 1/30:
            results['chemistry'] = (False, f"Химические связи слишком сильны: α={α:.4f} > 0.033")
        else:
            results['chemistry'] = (True, f"Химия возможна: α={α:.4f}")
        
        # 3. Образование углерода
        res_energy, res_quality = self.stellar.triple_alpha_resonance()
        if "Отличный" in res_quality or "Умеренный" in res_quality:
            results['carbon'] = (True, f"Углерод образуется: {res_quality}")
        else:
            results['carbon'] = (False, f"Углерод не образуется: {res_quality}")
        
        # 4. Энергия связи ядер
        binding_fe = self.nuclear.binding_energy_per_nucleon(56) / (self.u.const.e * 1e6)  # в МэВ
        if binding_fe < 0:
            results['nuclear'] = (False, f"Ядра нестабильны: E_связи = {binding_fe:.2f} МэВ")
        elif binding_fe < 1:
            results['nuclear'] = (False, f"Ядра слишком слабо связаны: {binding_fe:.2f} МэВ")
        else:
            results['nuclear'] = (True, f"Ядра стабильны: E_связи = {binding_fe:.2f} МэВ")
        
        # 5. Баланс сил
        α_G = self.grav.gravitational_coupling()
        ratio = α_G / α
        if ratio > 0.1:
            results['force_balance'] = (False, f"Гравитация доминирует: α_G/α = {ratio:.2e}")
        elif ratio < 1e-40:
            results['force_balance'] = (False, f"Гравитация слишком слаба для звезд: {ratio:.2e}")
        else:
            results['force_balance'] = (True, f"Баланс сил приемлем: α_G/α = {ratio:.2e}")
        
        # 6. Водородный синтез
        pp_rate = self.stellar.proton_proton_chain()
        if pp_rate['rate_relative'] < 0.01:
            results['fusion'] = (False, f"Термоядерный синтез слишком медленный")
        elif pp_rate['rate_relative'] > 100:
            results['fusion'] = (False, f"Синтез слишком быстрый, звезды быстро выгорают")
        else:
            results['fusion'] = (True, f"Синтез возможен, скорость ~{pp_rate['rate_relative']:.2f} от солнечной")
        
        return results
    
    def diagnose(self):
        """Выводит полный диагноз Вселенной"""
        results = self.analyze_all()
        
        print("\n" + "="*60)
        print(f"🔬 ПОЛНЫЙ ДИАГНОЗ ВСЕЛЕННОЙ: {self.u.name}")
        print("="*60)
        print(f"📊 Параметры:")
        print(f"   α (постоянная тонкой структуры) = {self.u.alpha:.6f}")
        print(f"   e/e₀ = {self.u.e/self.u.const.e:.3f}")
        print(f"   m_p/m_p₀ = {self.u.m_p/self.u.const.m_p:.3f}")
        print(f"   M_planck = {self.u.m_planck:.2e} кг")
        print(f"   L_planck = {self.u.l_planck:.2e} м")
        print()
        print(f"⚛️ Атомные свойства:")
        print(f"   Радиус Бора = {self.atomic.bohr_radius():.2e} м")
        print(f"   Энергия ионизации = {self.atomic.rydberg_ev():.2f} эВ")
        print(f"   Комптоновская длина = {self.atomic.compton_wavelength():.2e} м")
        print()
        print(f"☢️ Ядерные свойства:")
        print(f"   Масштаб КХД = {self.nuclear.qcd_scale():.2e} кг")
        print(f"   Энергия связи (Fe-56) = {self.nuclear.binding_energy_per_nucleon(56)/(self.u.const.e*1e6):.2f} МэВ/нуклон")
        print(f"   Кулоновский барьер (p+p) = {self.nuclear.coulomb_barrier(1,1)/(self.u.const.e*1e6):.2f} МэВ")
        print()
        print(f"🌟 Звездные процессы:")
        res_energy, res_quality = self.stellar.triple_alpha_resonance()
        print(f"   Тройная альфа: резонанс при {res_energy:.1f} кэВ, {res_quality}")
        pp_rate = self.stellar.proton_proton_chain()
        print(f"   pp-цепочка: скорость {pp_rate['rate_relative']:.2f} от солнечной")
        print()
        print(f"🌌 Гравитация:")
        print(f"   α_G = {self.grav.gravitational_coupling():.2e}")
        print(f"   m_p/M_planck = {self.grav.proton_to_planck_mass_ratio():.2e}")
        print()
        print(f"✅ ПРОВЕРКИ:")
        
        all_good = True
        for key, (passed, message) in results.items():
            icon = "✅" if passed else "❌"
            print(f"   {icon} {key.capitalize()}: {message}")
            all_good = all_good and passed
        
        print()
        if all_good:
            print("🎉 ВЕРДИКТ: ВСЕЛЕННАЯ ПРИГОДНА ДЛЯ ЖИЗНИ (как мы её знаем)!")
        else:
            print("💀 ВЕРДИКТ: БЕСПЛОДНАЯ ВСЕЛЕННАЯ")
        print("="*60)

class MultiverseExplorer:
    """Исследователь мультивселенной: создает и сравнивает разные реальности"""
    
    def __init__(self):
        self.universes = []
        
    def add_universe(self, universe: UniverseParameters):
        self.universes.append(universe)
        
    def create_universe_scan(self, param_name: str, values: List[float], base_universe: Optional[UniverseParameters] = None):
        """Создает серию вселенных, сканируя один параметр"""
        if base_universe is None:
            base_universe = UniverseParameters(name="Base")
        
        for i, val in enumerate(values):
            if param_name == "alpha":
                u = UniverseParameters(name=f"α={val:.4f}", alpha=val)
            elif param_name == "e":
                u = UniverseParameters(name=f"e/e₀={val:.2f}", e=val * base_universe.const.e)
            elif param_name == "m_p":
                u = UniverseParameters(name=f"m_p/m_p₀={val:.2f}", m_p=val * base_universe.const.m_p)
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            self.add_universe(u)
    
    def plot_properties_vs_alpha(self, property_func: Callable, ylabel: str, title: Optional[str] = None):
        """Строит график зависимости свойства от alpha"""
        alphas = []
        values = []
        
        for u in self.universes:
            alphas.append(u.alpha)
            analyzer = UniverseStabilityAnalyzer(u)
            values.append(property_func(analyzer))
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(alphas, values, 'bo-', markersize=8)
        plt.axvline(x=1/137.036, color='r', linestyle='--', label='Наша Вселенная (α≈1/137)')
        plt.xlabel('Постоянная тонкой структуры (α)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title if title else f'Зависимость {ylabel} от α', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.axvspan(1/300, 1/30, alpha=0.2, color='green', label='Зона возможной жизни')
        
        return plt
    
    def compare_all_universes(self):
        """Сравнивает все вселенные в мультивселенной"""
        print("\n" + "🔥"*30)
        print("МУЛЬТИВСЕЛЕННАЯ ОБСЕРВАТОРИЯ")
        print("🔥"*30)
        
        for i, u in enumerate(self.universes):
            print(f"\n[{i+1}] {u.name}")
            analyzer = UniverseStabilityAnalyzer(u)
            results = analyzer.analyze_all()
            
            score = sum(1 for passed, _ in results.values() if passed)
            print(f"    Оценка: {score}/{len(results)} критериев выполнено")
            if score == len(results):
                print("    🌟 ПОЛНОСТЬЮ ПРИГОДНА!")
            elif score > len(results)/2:
                print("    ✨ ЧАСТИЧНО ПРИГОДНА")
            else:
                print("    💀 НЕПРИГОДНА")

# ============= ДЕМОНСТРАЦИЯ =============

if __name__ == "__main__":
    # Создаем нашу Вселенную
    our_universe = UniverseParameters(name="🌍 Наша Вселенная")
    analyzer = UniverseStabilityAnalyzer(our_universe)
    analyzer.diagnose()
    
    # Создаем альтернативные вселенные с разными alpha
    explorer = MultiverseExplorer()
    
    alphas = [1/300, 1/200, 1/137.036, 1/100, 1/50, 1/30]
    for a in alphas:
        u = UniverseParameters(name=f"Мир α={a:.4f}", alpha=a)
        explorer.add_universe(u)
    
    explorer.compare_all_universes()
    
    # Строим графики
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    alphas_plot = np.logspace(-3, -1, 50)
    bohr_radii = []
    for a in alphas_plot:
        u = UniverseParameters(alpha=a)
        atomic = AtomicPhysics(u)
        bohr_radii.append(atomic.bohr_radius() / 5.29e-11)
    
    plt.semilogx(alphas_plot, bohr_radii)
    plt.axvline(x=1/137.036, color='r', linestyle='--', label='Наша Вселенная')
    plt.axvspan(1/300, 1/30, alpha=0.2, color='green', label='Зона жизни')
    plt.xlabel('α')
    plt.ylabel('Радиус атома (относительно нашей Вселенной)')
    plt.title('Зависимость размера атомов от α')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    binding_energies = []
    for a in alphas_plot:
        u = UniverseParameters(alpha=a)
        nuclear = NuclearPhysics(u)
        binding = nuclear.binding_energy_per_nucleon(56) / (8.5e6 * 1.602e-19)
        binding_energies.append(binding)
    
    plt.semilogx(alphas_plot, binding_energies)
    plt.axvline(x=1/137.036, color='r', linestyle='--', label='Наша Вселенная')
    plt.axvspan(1/300, 1/30, alpha=0.2, color='green', label='Зона жизни')
    plt.xlabel('α')
    plt.ylabel('Энергия связи (относительно нашей Вселенной)')
    plt.title('Зависимость ядерной энергии связи от α')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Экзотическая вселенная
    exotic = UniverseParameters(
        name="👽 Экзотическая Вселенная", 
        alpha=1/150,
        m_p=5 * UniversalConstants().m_p
    )
    analyzer_exotic = UniverseStabilityAnalyzer(exotic)
    analyzer_exotic.diagnose()
