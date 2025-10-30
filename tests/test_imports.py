try:
    from asc_itmo_lab.chaotic_measures import hurst_trajectory
    from asc_itmo_lab.enhanced_esn_fan import EnhancedESN_FAN

    print("✅ Все импорты работают корректно!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
