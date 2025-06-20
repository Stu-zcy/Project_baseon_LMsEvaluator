import gulp from 'gulp';
import open from 'gulp-open';
import browserSync from 'browser-sync';

const bs = browserSync.create(); // 创建一个 browser-sync 实例

const SRC_DIR = './';      
const SERVER_PORT = 5173;       // 服务器运行的端口
const START_PAGE = 'pages/sign-in.html'; // 服务器启动后默认打开的页面

// 任务：启动 BrowserSync 服务器并监视文件
function startServer(done) {
    bs.init({
        server: {
            baseDir: SRC_DIR
        },
        port: SERVER_PORT,
        open: 'local',   
				startPath: START_PAGE,                             
        notify: false,           // 关闭BrowserSync 通知
        // cors: true,           // 如果仍然有跨不同子域或端口的情况
    });

    // 监视文件变化
    gulp.watch(`./${SRC_DIR}/**/*.html`).on('change', bs.reload);
    gulp.watch(`./${SRC_DIR}/**/*.js`).on('change', bs.reload);
    // gulp.watch(`./${SRC_DIR}/**/*.css`).on('change', () => bs.reload("*.css")); // 或者 bs.stream()

    done();
}


function start() {
	gulp.src('pages/dashboard.html')
	.pipe(open());
}

// Task to open the browser
gulp.task('open-app', startServer);