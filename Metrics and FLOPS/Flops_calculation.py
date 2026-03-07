import tensorflow as tf

def calculate_flops_tf_profiler(model, batch_size=1):
    """
    Utilise le profiler TensorFlow pour calculer les FLOPS
    """
    # S'assurer que le modèle est construit
    if not model.built:
        model.build(input_shape=(batch_size,) + model.input_shape[1:])

    # Créer une fonction d'inférence
    @tf.function
    def infer(x):
        return model(x, training=False)

    # Obtenir le graphe concret
    concrete_func = infer.get_concrete_function(
        tf.TensorSpec([batch_size, *model.input_shape[1:]], model.input_dtype)
    )

    # Configurer le profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    try:
        # Profiler le graphe
        flops_stats = tf.compat.v1.profiler.profile(
            graph=concrete_func.graph,
            run_meta=run_meta,
            cmd='scope',
            options=opts
        )

        if flops_stats is not None:
            total_flops = flops_stats.total_float_ops
        else:
            total_flops = 0

        return total_flops

    except Exception as e:
        print(f"Erreur avec le profiler: {e}")
        print("Utilisation de la méthode manuelle...")
        return None

# Essayer d'abord avec le profiler
print("Tentative avec TensorFlow Profiler...")
flops_tf = calculate_flops_tf_profiler(model)

if flops_tf:
    print(f"FLOPS (TensorFlow Profiler): {flops_tf:,.0f}")
    print(f"GFLOPs: {flops_tf / 1e9:.2f}")
else:
    # Fallback sur la méthode manuelle
    print("\nUtilisation de la méthode manuelle...")
    total_flops, details = calculate_flops_accurate(model)