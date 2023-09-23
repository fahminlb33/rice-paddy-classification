import { useEffect, useRef, useState } from 'react'
import { Button, Container, FileButton, Flex, Grid, Image, LoadingOverlay, Tabs, Text, Title } from "@mantine/core";
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { IconPhoto, IconPhotoFilled, IconPhotoHexagon } from '@tabler/icons-react';

import * as utils from './ml/utils'
import * as predictor from './ml/predict'

import styles from './App.module.css'

const CLASS_CATEGORIES = {
  0: 'Bacterial Blight',
  1: 'Blast',
  2: 'Brown Spot',
  3: 'Healthy',
  4: 'Tungro',
}

const SAMPLE_IMAGES = [
  "/samples/bacterial_blight-1.jpg",
  "/samples/bacterial_blight-2.jpg",
  "/samples/blast-1.jpg",
  "/samples/blast-2.jpg",
  "/samples/brown_spot-1.jpg",
  "/samples/brown_spot-2.jpg",
  "/samples/tungro-1.jpg",
  "/samples/tungro-2.jpg",
  "/samples/healthy-1.jpg",
  "/samples/healthy-2.jpg",
]

export default function App() {
  const [loadingVisible, setLoadingVisible] = useState(true);
  const [model, setModel] = useState(null);
  const originalImageRef = useRef(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [superimposedImage, setSuperimposedImage] = useState(null);
  const [clippedImage, setClippedImage] = useState(null);
  const [chartProba, setChartProba] = useState(null);
  const [threshold, setThreshold] = useState(0);

  async function upload(uploadedImage) {
    // set original image
    setOriginalImage((old) => {
      if (old) URL.revokeObjectURL(old)
      return typeof uploadedImage !== "string" ? URL.createObjectURL(uploadedImage) : uploadedImage
    });
  }

  async function predict() {
    // predict
    const img = predictor.loadImage(originalImageRef.current)
    const proba = await predictor.predict(model, img)
    const predictedClass = await predictor.getClass(proba)

    // create Grad-CAM
    const gradCamTensor = predictor.gradCAM(model, img, predictedClass);
    const superimposedImage = await predictor.superimposeImage(img, gradCamTensor)
    const superimposedBlob = await utils.tensorToBlob(superimposedImage);
    const maskedImage = await predictor.maskImage(img, gradCamTensor)
    const maskedBlob = await utils.tensorToBlob(maskedImage);

    // calculate threshold
    const thr = await predictor.getThreshold(gradCamTensor);

    // map class probabilities to data
    const chartProba = proba.map((p, i) => {
      return {
        category: CLASS_CATEGORIES[i],
        probability: p * 100,
        highest: i === predictedClass
      }
    });
    chartProba.push(chartProba.splice(3, 1)[0])

    // set states
    setThreshold(thr);
    setChartProba(chartProba)
    setSuperimposedImage((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(superimposedBlob)
    });
    setClippedImage((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(maskedBlob)
    });
  }

  useEffect(() => {
    console.log("Loading model...")
    predictor.loadModel().then(m => {
      setModel(m)
      setLoadingVisible(false)
      console.log("Model loaded")
    })
  }, []);

  return (
    <Container className={styles.container_main}>
      <LoadingOverlay visible={loadingVisible} zIndex={1000} overlayProps={{ radius: "sm", blur: 2 }} />
      
      <Title>Padi-CNN: Rice disease classification</Title>
      <Text size='sm'>MobileNetV2-based classifier with Grad-CAM visualization</Text>

      <Grid gutter="xl" className={styles.container_tabs}>
        <Grid.Col span={4}>
          {/* Image upload */}
          <Title order={4}>Upload image</Title>
          <FileButton accept='image/png,image/jpeg' onChange={upload}>
            {(props) => <Button fullWidth {...props}>Upload</Button>}
          </FileButton>

          <Text mt="xs">Sample images:</Text>
          <Flex gap="sm" wrap="wrap">
            {SAMPLE_IMAGES.map(url => (
              <Image key={url} className={styles.image_thumbnails} src={url} onClick={() => upload(url)} />
            ))}
          </Flex>

          {/* Classification results */}
          <Title order={4} mt="lg">Classification results</Title>
          <Text mb="sm">Threshold: {threshold.toFixed(4)}</Text>
          <div className={styles.container_chart}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartProba} layout='vertical' margin={{ left: 20, right: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" label={{ value: 'Probability %', position: "bottom" }} />
                <YAxis dataKey="category" type='category' />
                <Tooltip formatter={x => x.toFixed(2) + "%"} />

                <Bar dataKey="probability">
                  {chartProba && chartProba.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.highest ? '#8884d8' : '#82ca9d'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Grid.Col>

        {/* Image preview */}
        <Grid.Col span={8}>
          <Title order={4}>Processed image</Title>
          <Tabs defaultValue="original" >
            <Tabs.List>
              <Tabs.Tab value="original" leftSection={<IconPhoto className={styles.tab_icon} />}>
                Original
              </Tabs.Tab>
              <Tabs.Tab value="superimposed" leftSection={<IconPhotoFilled className={styles.tab_icon} />}>
                Superimposed
              </Tabs.Tab>
              <Tabs.Tab value="clipped" leftSection={<IconPhotoHexagon className={styles.tab_icon} />}>
                Clipped
              </Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="original" p="md">
              <Image radius="md" className={styles.image_original} src={originalImage} ref={originalImageRef} onLoad={predict} />
            </Tabs.Panel>
            <Tabs.Panel value="superimposed" p="md">
              <Image radius="md" src={superimposedImage} />
            </Tabs.Panel>
            <Tabs.Panel value="clipped" p="md">
              <Image radius="md" src={clippedImage} />
            </Tabs.Panel>
          </Tabs>
        </Grid.Col>
      </Grid>
    </Container>
  )
}
