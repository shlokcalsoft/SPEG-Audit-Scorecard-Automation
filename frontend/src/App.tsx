import { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileSpreadsheet, Folder, Loader2, Moon, Sun } from 'lucide-react';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [projects, setProjects] = useState<any[]>([]);
  const [selectedProject, setSelectedProject] = useState('default');
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Toggle dark mode by adding/removing a class on the html body or main wrapper
  // We'll manage it locally in this wrapper component using tailwind's dark: modifiers
  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  const fetchProjects = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/projects');
      setProjects(response.data.projects);
      if (response.data.projects.length > 0) {
        setSelectedProject(response.data.projects[0].id || response.data.projects[0].name);
      }
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return alert('Please select a file first.');
    
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('project', selectedProject);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        responseType: 'blob'
      });
      
      // Trigger download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${file.name.replace('.docx', '')}_Audit_Completed.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
    } catch (error) {
      console.error(error);
      alert('Error processing the file. Check server logs.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={`${isDarkMode ? 'dark' : ''}`}>
      <div className="min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex transition-colors duration-200">
        
        {/* Sidebar */}
        <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col transition-colors duration-200">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h1 className="text-xl font-bold flex items-center gap-2">
              <FileSpreadsheet className="text-blue-600 dark:text-blue-400" />
              SEPG Audit Automation
            </h1>
          </div>
          <nav className="flex-1 p-4 space-y-2">
            <button 
              onClick={() => setActiveTab('dashboard')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                activeTab === 'dashboard' 
                  ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 font-medium' 
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Upload size={20} />
              Dashboard
            </button>
            <button 
              onClick={() => { setActiveTab('projects'); fetchProjects(); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                activeTab === 'projects' 
                  ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 font-medium' 
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Folder size={20} />
              Projects
            </button>
          </nav>
          
          {/* Dark Mode Toggle at the bottom of sidebar */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={toggleDarkMode}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
              {isDarkMode ? 'Light Mode' : 'Dark Mode'}
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8">
          {activeTab === 'dashboard' && (
            <div className="max-w-2xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-8 transition-colors duration-200">
              <h2 className="text-2xl font-bold mb-6">Upload Transcript</h2>
              
              <div className="mb-6">
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">Select Project Template</label>
                <select 
                  value={selectedProject}
                  onChange={(e) => setSelectedProject(e.target.value)}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-3 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {projects.map((proj, idx) => (
                    <option key={idx} value={proj.id || proj.name}>
                      {proj.name} ({proj.folder})
                    </option>
                  ))}
                </select>
              </div>

              <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-12 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors bg-gray-50 dark:bg-gray-900/50">
                <input 
                  type="file" 
                  id="file-upload" 
                  className="hidden" 
                  accept=".docx"
                  onChange={handleFileChange}
                />
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                  <Upload size={48} className="text-gray-400 mb-4" />
                  <span className="text-gray-600 dark:text-gray-300 text-lg mb-2">
                    {file ? file.name : 'Click to select transcript (.docx)'}
                  </span>
                  <span className="text-sm text-gray-400">Supported format: DOCX</span>
                </label>
              </div>
              
              <button 
                onClick={handleUpload}
                disabled={isProcessing || !file}
                className={`mt-6 w-full py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-white font-medium transition-all
                  ${isProcessing || !file ? 'bg-blue-300 dark:bg-blue-800 cursor-not-allowed text-gray-100 dark:text-gray-300' : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 shadow-md hover:shadow-lg'}`}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="animate-spin" /> Processing Pipeline...
                  </>
                ) : (
                  'Generate Scorecard'
                )}
              </button>
            </div>
          )}

          {activeTab === 'projects' && (
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl font-bold mb-6">Available Projects</h2>
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden transition-colors duration-200">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-gray-50 dark:bg-gray-700/50 border-b border-gray-200 dark:border-gray-700">
                      <th className="p-4 text-sm font-semibold text-gray-600 dark:text-gray-300">Project Name</th>
                      <th className="p-4 text-sm font-semibold text-gray-600 dark:text-gray-300">Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {projects.length > 0 ? (
                      projects.map((proj, idx) => (
                        <tr key={idx} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                          <td className="p-4 flex items-center gap-3 text-gray-800 dark:text-gray-200">
                            <FileSpreadsheet size={18} className="text-blue-600 dark:text-blue-400" />
                            {proj.name}
                          </td>
                          <td className="p-4 text-gray-600 dark:text-gray-400 capitalize">
                            <span className="bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full text-xs font-medium border border-gray-200 dark:border-gray-600">
                              {proj.folder}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={2} className="p-8 text-center text-gray-500 dark:text-gray-400">
                          No projects found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
