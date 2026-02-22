#version 150

in vec2 v_texcoord;
out vec4 fragColor;

uniform float time;
uniform float intensity;
uniform float fade;

void main() {
    vec2 uv = v_texcoord - 0.5;
    float r = length(uv);
    float phi = atan(uv.y, uv.x);

    float spiral = sin(phi * 5.0 - log(r + 0.001) * 8.0 + time * 3.0);
    spiral = 0.5 + 0.5 * spiral;

    float spiral2 = sin(phi * 3.0 + log(r + 0.001) * 6.0 - time * 2.0);
    spiral2 = 0.5 + 0.5 * spiral2;

    float pattern = spiral * 0.6 + spiral2 * 0.4;

    vec3 color = vec3(
        0.5 + 0.5 * sin(phi + time * 0.7 + r * 4.0),
        0.5 + 0.5 * sin(phi + time * 0.7 + r * 4.0 + 2.094),
        0.5 + 0.5 * sin(phi + time * 0.7 + r * 4.0 + 4.189)
    );

    color *= pattern;

    float alpha = intensity * fade * smoothstep(0.6, 0.05, r) * 0.5;

    fragColor = vec4(color, alpha);
}
