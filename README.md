1. use default shader and window

   ```
   #include <GWindow.h>
   Eigen::MatrixXd V;
   Eigen::MatrixXi F;
   igl::read_triangle_mesh("cube.obj", V, F);
   YRender::GWindow window("window");
   window.push_object(V, F, YRender::Y_TRIANGLES);
   window.show();
   ```

2. use user-defined shader

   ```
   //example
   window.set_shader("path of vertex shader","path of fragment shader","path of geometry Shader");
   //set shader uniform value
   window.set_shader_attri("atti_name",value);
   ```

   

3. use user-defined interaction

   ```
   class MyWindow:public YRender::GWindow {
   protected:
   	void mouse_button_callback(GLFWwindow* window, int button, int action, int modifiers) override;
   	void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) override;
   	void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods)override;
   	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)override;
   	void framebuffer_size_callback(GLFWwindow* window, int width, int height) override;
   }
   void MyWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int modifiers)
   {
   	GWindow::mouse_button_callback(window, button, action, modifiers);	
   	//add your code
   }
   ```

   