#version 150

in vec2 v_texcoord;
out vec4 fragColor;

uniform float time;
uniform float intensity;
uniform float fade;

void main() {
    vec2 uv = v_texcoord;

    float zoom = 2.0 + 1.5 * sin(time * 0.15);
    vec2 center = vec2(-0.745, 0.186);
    vec2 c = center + (uv - 0.5) / zoom;

    float jt = time * 0.2;
    vec2 julia_c = vec2(-0.7 + 0.2 * sin(jt), 0.27 + 0.15 * cos(jt * 1.3));
    vec2 z = (uv - 0.5) * 3.0 / zoom;

    int max_iter = 64;
    int iter = 0;
    for (int i = 0; i < 64; i++) {
        if (dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + julia_c;
        iter++;
    }

    float t_color = float(iter) / float(max_iter);
    t_color = sqrt(t_color);

    vec3 color = vec3(
        0.5 + 0.5 * sin(t_color * 6.28 + time * 1.0),
        0.5 + 0.5 * sin(t_color * 6.28 + time * 1.0 + 2.094),
        0.5 + 0.5 * sin(t_color * 6.28 + time * 1.0 + 4.189)
    );

    if (iter == max_iter) {
        color = vec3(0.0, 0.0, 0.05 + 0.05 * sin(time * 3.0));
    }

    float alpha = intensity * fade * 0.6;
    fragColor = vec4(color, alpha);
}
