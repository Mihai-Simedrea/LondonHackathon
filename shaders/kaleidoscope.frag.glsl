#version 150

in vec2 v_texcoord;
out vec4 fragColor;

uniform float time;
uniform float intensity;
uniform float fade;
uniform float segments;

void main() {
    vec2 uv = v_texcoord - 0.5;
    float r = length(uv);

    float phi = atan(uv.y, uv.x);
    float seg = 2.0 * 3.14159265 / segments;

    phi = mod(phi + time * 0.3, seg);
    phi = abs(phi - seg * 0.5);

    float radial_wave = sin(r * 15.0 - time * 4.0);
    float radial_ring = 0.5 + 0.5 * radial_wave;

    float petal = 0.5 + 0.5 * cos(phi * segments * 0.5);
    float expand = 0.15 + 0.25 * (0.5 + 0.5 * sin(time * 1.2));
    float flower = smoothstep(expand + 0.05, expand - 0.05, abs(r - 0.2 - expand * 0.5) - petal * expand);

    float rings = 0.5 + 0.5 * sin(r * 20.0 - time * 5.0 + phi * 2.0);

    float pattern = max(flower * 0.8, rings * radial_ring * 0.5);

    float hue = r * 8.0 - time * 2.0 + phi * 1.5;
    vec3 color = vec3(
        0.5 + 0.5 * sin(hue),
        0.5 + 0.5 * sin(hue + 2.094),
        0.5 + 0.5 * sin(hue + 4.189)
    );

    color *= 0.4 + pattern * 1.2;

    float edge_fade = smoothstep(0.55, 0.1, r);
    float alpha = intensity * fade * edge_fade * (0.15 + pattern * 0.7);

    fragColor = vec4(color, alpha);
}
