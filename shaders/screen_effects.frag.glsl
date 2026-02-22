#version 150

in vec2 v_texcoord;
out vec4 fragColor;

uniform float time;
uniform float intensity;
uniform float fade;
uniform float effect_type;  // 0=dim, 1=flicker, 2=inversion, 3=chromatic

void main() {
    vec2 uv = v_texcoord;
    float alpha = intensity * fade;

    if (effect_type < 0.5) {
        // DIM
        float dim_level = 0.3 + 0.15 * sin(time * 1.2);
        fragColor = vec4(0.0, 0.0, 0.0, alpha * dim_level);

    } else if (effect_type < 1.5) {
        // FLICKER
        float flicker = 0.5 + 0.5 * sin(time * 25.0 + sin(time * 7.0) * 5.0);
        flicker *= 0.5 + 0.5 * sin(time * 37.0);
        fragColor = vec4(vec3(flicker * 0.3), alpha * 0.5);

    } else if (effect_type < 2.5) {
        // COLOR INVERSION pulse
        float pulse = 0.5 + 0.5 * sin(time * 2.0);
        vec3 tint = vec3(0.8, 0.2, 0.9) * pulse;
        fragColor = vec4(tint, alpha * 0.3);

    } else {
        // CHROMATIC ABERRATION hint
        float edge = smoothstep(0.3, 0.5, length(uv - 0.5));
        vec3 color = vec3(
            edge * (0.5 + 0.5 * sin(time * 3.0)),
            edge * (0.5 + 0.5 * sin(time * 3.0 + 2.0)),
            edge * (0.5 + 0.5 * sin(time * 3.0 + 4.0))
        );
        fragColor = vec4(color, alpha * edge * 0.4);
    }
}
