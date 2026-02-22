#version 150

in vec2 v_texcoord;
out vec4 fragColor;

uniform float time;
uniform float intensity;
uniform float fade;

float circle(vec2 p, vec2 center, float radius) {
    return abs(length(p - center) - radius);
}

void main() {
    vec2 uv = v_texcoord - 0.5;

    float radius = 0.12 + 0.02 * sin(time * 1.5);
    float line_w = 0.004;

    float d = 1.0;

    d = min(d, circle(uv, vec2(0.0), radius));

    for (int i = 0; i < 6; i++) {
        float angle = float(i) * 3.14159265 / 3.0 + time * 0.2;
        vec2 center = vec2(cos(angle), sin(angle)) * radius;
        d = min(d, circle(uv, center, radius));
    }

    for (int i = 0; i < 6; i++) {
        float angle = float(i) * 3.14159265 / 3.0 + 3.14159265 / 6.0 + time * 0.2;
        vec2 center = vec2(cos(angle), sin(angle)) * radius * 1.732;
        d = min(d, circle(uv, center, radius));
    }

    float glow = smoothstep(line_w * 3.0, 0.0, d);
    float line = smoothstep(line_w, line_w * 0.3, d);

    float hue = atan(uv.y, uv.x) + time * 0.5;
    vec3 color = vec3(
        0.5 + 0.5 * sin(hue),
        0.5 + 0.5 * sin(hue + 2.094),
        0.5 + 0.5 * sin(hue + 4.189)
    );

    vec3 final_color = color * (line * 0.9 + glow * 0.4);

    float alpha = intensity * fade * (line * 0.8 + glow * 0.3);
    fragColor = vec4(final_color, alpha);
}
