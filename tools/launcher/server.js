// tools/launcher/server.js
import express from 'express';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs'; 

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000; // 启动器 UI 端口

// [配置] 模型所在的固定目录
const MODELS_DIR = '/root/autodl-tmp/models/sparkinfer';

app.use(express.json());
app.use(express.static(__dirname));

const EXECUTABLE_PATH = path.resolve(__dirname, '../../build_rel/bin/llama-server');
let serverProcess = null;

// [接口] 获取模型文件列表
app.get('/api/files', (req, res) => {
    try {
        if (!fs.existsSync(MODELS_DIR)) {
            return res.json({ files: [] }); 
        }
        
        const files = fs.readdirSync(MODELS_DIR).filter(file => {
            return file.endsWith('.gguf'); // 只列出 .gguf 文件
        });
        
        res.json({ files });
    } catch (e) {
        console.error("读取目录失败:", e);
        res.status(500).json({ error: "无法读取模型目录" });
    }
});

app.post('/start-server', (req, res) => {
    let { model, model_split, vram_budget, threads } = req.body;

    // 路径拼接逻辑
    if (model && !model.startsWith('/')) {
        model = path.join(MODELS_DIR, model);
    }
    if (model_split && !model_split.startsWith('/')) {
        model_split = path.join(MODELS_DIR, model_split);
    }

    // [需求 1 实现] 如果服务已经在运行，直接返回成功和跳转链接，不再报错
    if (serverProcess) {
        console.log('检测到服务已在运行，返回现有地址。');
        return res.json({ 
            message: '服务已在运行，准备跳转...', 
            targetUrl: 'http://localhost:8080' 
        });
    }

    console.log('--- 收到启动请求 ---');
    console.log(`Model: ${model}`);
    console.log(`Split: ${model_split}`);
    console.log(`VRAM:  ${vram_budget}`);
    console.log(`Threads: ${threads}`);

    const envVars = {
        ...process.env,
        SPIF_DFR_EMA: 'ON',
        SPIF_INIT_DFR_DECAY: '67',
        SPIF_DX_DFR_DECAY: '51',
        SPIF_RELOAD_WINDOW_SIZE: '4',
        SPIF_REORDER: 'ON',
        SPIF_PARALLEL: 'ON',
        SPIF_RELOAD: 'ON'
    };

    const args = [
        '-m', model,                
        '-spif-ms', model_split,    
        '-vb', String(vram_budget), 
        '-t', String(threads),      
        '-cffn',                    
        '--no-mmap',                
        '-s', '1234',               
        '-c', '2048',               
        '--port', '8080',           
        '--host', '0.0.0.0'         
    ];

    console.log(`Executing: ${EXECUTABLE_PATH} ${args.join(' ')}`);

    try {
        serverProcess = spawn(EXECUTABLE_PATH, args, { env: envVars });

        serverProcess.stdout.on('data', (data) => {
            const output = data.toString();
            console.log(`[SparkInfer]: ${output}`);
        });

        serverProcess.stderr.on('data', (data) => {
            console.error(`[SparkInfer Log]: ${data}`);
        });

        // [需求 2 核心] 监听进程错误（如启动失败）
        serverProcess.on('error', (err) => {
            console.error('[SparkInfer] 启动失败:', err);
            serverProcess = null; // 重置状态，允许前端再次尝试
        });

        // [需求 2 核心] 监听进程退出
        serverProcess.on('close', (code) => {
            console.log(`SparkInfer process exited with code ${code}`);
            serverProcess = null; // 重置状态，允许前端再次尝试
        });

        // 立即返回成功，前端接到 targetUrl 后会处理跳转
        res.json({ message: '正在启动 SparkInfer...', targetUrl: 'http://localhost:8080' });

    } catch (e) {
        console.error("Launch Failed:", e);
        serverProcess = null; // 确保异常时重置状态
        res.status(500).json({ error: e.message });
    }
});

app.listen(PORT, () => {
    console.log('=================================================');
    console.log(`Launcher UI running at http://localhost:${PORT}`);
    console.log('Ensure you have SSH tunneling set up:');
    console.log('ssh -L 3000:localhost:3000 -L 8080:localhost:8080 ...');
    console.log('=================================================');
});