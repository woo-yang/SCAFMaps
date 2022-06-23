#version 330 core
out vec4 FragColor;


struct Lighting {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};  


in vec3 FragPos;  
in vec3 Normal;  

uniform vec3 objColor;
uniform vec3 viewPos;
uniform Lighting dirLight;

vec3 CalcDirLight(Lighting light, vec3 normal, vec3 viewDir,vec3 objColor);


void main()
{    
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    vec3 result = vec3(0.f);


    result += CalcDirLight(dirLight, norm, viewDir,objColor);  
    FragColor = vec4(result, 1.0);
    
    //FragColor = vec4(1.0);
    
}

vec3 CalcDirLight(Lighting light, vec3 normal, vec3 viewDir,vec3 objColor)
{
    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 500);

    vec3 ambient = light.ambient * objColor;
    vec3 diffuse = light.diffuse * diff * objColor;
    vec3 specular = light.specular * spec * objColor;
    return (ambient + diffuse + specular);
}


