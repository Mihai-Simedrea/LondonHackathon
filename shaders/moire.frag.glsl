#version 150

in vec2 v_texcoord;
out vec4 fragColor;

uniform float time;
uniform float intensity;
uniform float fade;

void main() {
    vec2 uv = v_texcoord;

    float freq1 = 30.0 + 5.0 * sin(time * 0.3);
    float freq2 = 31.5 + 5.0 * cos(time * 0.4);
    float angle_offset = 0.05 + 0.03 * sin(time * 0.2);

    float grid1 = sin(uv.x * freq1 + time * 2.0) * sin(uv.y * freq1 - time * 1.5);

    vec2 rotated = vec2(
        uv.x * cos(angle_offset) - uv.y * sin(angle_offset),
        uv.x * sin(angle_offset) + uv.y * cos(angle_offset)
    );
    float grid2 = sin(rotated.x * freq2 - time * 1.8) * sin(rotated.y * freq2 + time * 1.2);

    float pattern = grid1 * grid2;
    pattern = 0.5 + 0.5 * pattern;

    float dist = length(uv - 0.5);
    vec3 color = vec3(
        0.5 + 0.5 * sin(dist * 20.0 + time + pattern * 3.0),
        0.5 + 0.5 * sin(dist * 20.0 + time + pattern * 3.0 + 2.094),
        0.5 + 0.5 * sin(dist * 20.0 + time + pattern * 3.0 + 4.189)
    );

    color *= pattern;

    float alpha = intensity * fade * 0.35;
    fragColor = vec4(color, alpha);
}
