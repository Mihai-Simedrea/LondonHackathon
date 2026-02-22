#version 150

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 v_texcoord;

uniform mat4 p3d_ModelViewProjectionMatrix;

void main() {
    v_texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
