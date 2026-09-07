import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config. 90s timeout because some
// requests (e.g. AI Copilot's Claude calls) can legitimately take a while.
const api: AxiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 90000,
});

// Request interceptor
api.interceptors.request.use(
    (config) => {
        // Add auth token to requests
        const token = localStorage.getItem('auth_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error: AxiosError) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
    (response) => response,
    async (error: AxiosError) => {
        if (error.response) {
            // Server responded with error status
            const { status, data } = error.response;

            switch (status) {
                case 401:
                    // Handle unauthorized - redirect to login
                    console.error('Unauthorized access');
                    localStorage.removeItem('auth_token');
                    if (!window.location.pathname.includes('/login')) {
                        window.location.href = '/login';
                    }
                    break;
                case 403:
                    console.error('Forbidden access');
                    break;
                case 404:
                    console.error('Resource not found');
                    break;
                case 429:
                    console.error('Too many requests - rate limited');
                    break;
                case 500:
                    console.error('Internal server error');
                    break;
                default:
                    console.error(`API error: ${status}`);
            }

            // Extract error message safely from various backend error formats
            let errorMessage = 'An error occurred';
            if ((data as any)?.error?.message && typeof (data as any).error.message === 'string') {
                errorMessage = (data as any).error.message;
            } else if (typeof (data as any)?.detail === 'string') {
                errorMessage = (data as any).detail;
            } else if (Array.isArray((data as any)?.detail) && (data as any).detail.length > 0) {
                errorMessage = (data as any).detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
            } else if (typeof (data as any)?.message === 'string') {
                errorMessage = (data as any).message;
            }

            // Return structured error
            return Promise.reject({
                status,
                message: errorMessage,
                data
            });
        } else if (error.request) {
            // Request made but no response received
            console.error('No response from server');
            const isTimeout = error.code === 'ECONNABORTED' || error.message?.includes('timeout');
            return Promise.reject({
                status: 0,
                message: isTimeout
                    ? 'Connection timed out. Please try again.'
                    : 'Network error - please check that the backend is running and reachable.'
            });
        } else {
            // Something went wrong setting up the request
            console.error('Request setup error:', error.message);
            return Promise.reject({
                status: -1,
                message: error.message
            });
        }
    }
);

export default api;

// Export types for use in services
export type { AxiosError, AxiosRequestConfig };
