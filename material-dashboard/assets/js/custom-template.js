// --- Sidenav Toggler Logic ---

(function() {
  // 查找需要操作的元素
  var iconNavbarSidenav = document.getElementById('iconNavbarSidenav');
  var sidenav = document.getElementById('sidenav-main'); // 侧边栏本身，ID通常是这个
  var body = document.getElementsByTagName('body')[0];

  // 定义侧边栏展开时 body 上会添加的 class
  // 在 Argon / Material Dashboard 这类主题中，通常是 'g-sidenav-pinned'。需要检查一下。
  var bodyOpenClassName = 'g-sidenav-pinned'; 
  
  // 检查元素是否存在，防止报错
  if (iconNavbarSidenav && sidenav && body) {
    iconNavbarSidenav.addEventListener('click', function() {
      // 切换 body 上的 class，这个 class 控制主内容区域的边距
      body.classList.toggle(bodyOpenClassName);
      
      // 你也可以在这里直接控制 sidenav 的 class，但通常控制 body 就足够了
      // 如果需要，可以取消下面这行注释并替换为正确的 class
      // sidenav.classList.toggle('open-class-for-sidenav'); 
    });
  }

  var sidenavMain = document.querySelector('.sidenav-main'); // 根据你的HTML结构调整
  if (sidenavMain) {
    sidenavMain.addEventListener('click', function(e) {
      // 如果点击的不是菜单项，并且侧边栏是打开的，则关闭它
      if (!e.target.closest('.navbar-nav') && body.classList.contains(bodyOpenClassName)) {
        body.classList.remove(bodyOpenClassName);
      }
    });
  }
})();
