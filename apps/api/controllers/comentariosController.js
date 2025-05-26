import supabase from '../DB/supabaseClient.js';

// Obtener comentarios
export const obtenerComentarios = async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('comentarios')
      .select('id, texto, creado_en, usuarios (usuario, nombres)')
      .order('creado_en', { ascending: false });

    if (error) throw error;

    // Formatea los comentarios para mostrar nombre o usuario
    const comentarios = data.map(c => ({
      id: c.id,
      texto: c.texto,
      usuario: c.usuarios?.nombres || c.usuarios?.usuario || 'Anónimo',
      creado_en: c.creado_en
    }));

    res.json(comentarios);
  } catch (err) {
    console.error('Error al obtener comentarios:', err);
    res.status(500).json({ error: 'Error al obtener comentarios' });
  }
};

// Crear comentario
export const crearComentario = async (req, res) => {
  const { texto, usuario_id } = req.body;
  if (!texto) return res.status(400).json({ error: 'Texto requerido' });

  try {
    const { data, error } = await supabase
      .from('comentarios')
      .insert([{ texto, usuario_id }])
      .select()
      .single();

    if (error) throw error;

    res.status(201).json(data);
  } catch (err) {
    console.error('Error al guardar comentario:', err);
    res.status(500).json({ error: 'Error al guardar comentario' });
  }
};